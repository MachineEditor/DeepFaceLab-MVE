import math
from pathlib import Path

import cv2
import numpy as np
import numpy.linalg as npla


class DeepPortraitRelighting(object):
    
    def __init__(self):
        from nnlib import nnlib
        nnlib.import_torch()        
        self.torch = nnlib.torch
        self.torch_device = nnlib.torch_device        
        self.model = DeepPortraitRelighting.build_model(self.torch, self.torch_device)

    def SH_basis(self, alt, azi):
        alt = alt * math.pi / 180.0
        azi = azi * math.pi / 180.0
        
        x = math.cos(alt)*math.sin(azi)
        y = -math.cos(alt)*math.cos(azi)        
        z = math.sin(alt)
         
        normal = np.array([x,y,z])
        
        norm_X = normal[0]
        norm_Y = normal[1]
        norm_Z = normal[2]

        sh_basis = np.zeros((9))
        att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
        sh_basis[0] = 0.5/np.sqrt(np.pi)*att[0]

        sh_basis[1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
        sh_basis[2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
        sh_basis[3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

        sh_basis[4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
        sh_basis[5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
        sh_basis[6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
        sh_basis[7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
        sh_basis[8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
        return sh_basis
        
    #n = [0..8]
    def relight(self, img, alt, azi, intensity=1.0, lighten=False):
        torch = self.torch
    
        sh = self.SH_basis (alt, azi)   
        sh = (sh.reshape( (1,9,1,1) ) ).astype(np.float32)      
        #sh *= 0.1  
        sh = torch.autograd.Variable(torch.from_numpy(sh).to(self.torch_device))
        
        row, col, _ = img.shape
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:,:,0]
        outputImg, outputSH  = self.model(torch.autograd.Variable(torch.from_numpy(inputL[None,None,...].astype(np.float32)/255.0).to(self.torch_device)), 
                                          sh, 0)
        
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = np.clip (outputImg, 0.0, 1.0)        
        outputImg = cv2.blur(outputImg, (3,3) ) 

        if not lighten:
            outputImg = inputL*(1.0-intensity) + (inputL*outputImg)*intensity
        else:
            outputImg = inputL*(1.0-intensity) + (outputImg*255.0)*intensity
            
        outputImg = np.clip(outputImg, 0,255).astype(np.uint8)   

        Lab[:,:,0] = outputImg
        result = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        result = cv2.resize(result, (col, row))
        return result

    @staticmethod
    def build_model(torch, torch_device):
        nn = torch.nn
        F = torch.nn.functional

        def conv3X3(in_planes, out_planes, stride=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
            
        # define the network
        class BasicBlock(nn.Module):
            def __init__(self, inplanes, outplanes, batchNorm_type=0, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                # batchNorm_type 0 means batchnormalization
                #                1 means instance normalization
                self.inplanes = inplanes
                self.outplanes = outplanes
                self.conv1 = conv3X3(inplanes, outplanes, 1)
                self.conv2 = conv3X3(outplanes, outplanes, 1)
                if batchNorm_type == 0:
                    self.bn1 = nn.BatchNorm2d(outplanes)
                    self.bn2 = nn.BatchNorm2d(outplanes)
                else:
                    self.bn1 = nn.InstanceNorm2d(outplanes)
                    self.bn2 = nn.InstanceNorm2d(outplanes)
                
                self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
            
            def forward(self, x):
                out = self.conv1(x)
                out = self.bn1(out)
                out = F.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.inplanes != self.outplanes:
                    out += self.shortcuts(x)
                else:
                    out += x
                
                out = F.relu(out)
                return out

        class HourglassBlock(nn.Module):
            def __init__(self, inplane, mid_plane, middleNet, skipLayer=True):
                super(HourglassBlock, self).__init__()
                # upper branch
                self.skipLayer = True
                self.upper = BasicBlock(inplane, inplane, batchNorm_type=1)
                
                # lower branch
                self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
                self.low1 = BasicBlock(inplane, mid_plane)
                self.middle = middleNet
                self.low2 = BasicBlock(mid_plane, inplane, batchNorm_type=1)

            def forward(self, x, light, count, skip_count):
                # we use count to indicate wich layer we are in
                # max_count indicates the from which layer, we would use skip connections
                out_upper = self.upper(x)
                out_lower = self.downSample(x)
                out_lower = self.low1(out_lower)
                out_lower, out_middle = self.middle(out_lower, light, count+1, skip_count)
                out_lower = self.low2(out_lower)                
                out_lower = self.upSample(out_lower)            
                if count >= skip_count and self.skipLayer:
                    out = out_lower + out_upper
                else:
                    out = out_lower
                return out, out_middle

        class lightingNet(nn.Module):
            def __init__(self, ncInput, ncOutput, ncMiddle):
                super(lightingNet, self).__init__()
                self.ncInput = ncInput
                self.ncOutput = ncOutput
                self.ncMiddle = ncMiddle
                self.predict_FC1 = nn.Conv2d(self.ncInput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
                self.predict_relu1 = nn.PReLU()
                self.predict_FC2 = nn.Conv2d(self.ncMiddle, self.ncOutput, kernel_size=1, stride=1, bias=False)

                self.post_FC1 = nn.Conv2d(self.ncOutput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
                self.post_relu1 = nn.PReLU()
                self.post_FC2 = nn.Conv2d(self.ncMiddle, self.ncInput, kernel_size=1, stride=1, bias=False)
                self.post_relu2 = nn.ReLU()  # to be consistance with the original feature

            def forward(self, innerFeat, target_light, count, skip_count):            
                x = innerFeat[:,0:self.ncInput,:,:] # lighting feature
                _, _, row, col = x.shape
                # predict lighting
                feat = x.mean(dim=(2,3), keepdim=True)            
                light = self.predict_relu1(self.predict_FC1(feat))
                light = self.predict_FC2(light)
                upFeat = self.post_relu1(self.post_FC1(target_light))
                upFeat = self.post_relu2(self.post_FC2(upFeat))
                upFeat = upFeat.repeat((1,1,row, col))
                innerFeat[:,0:self.ncInput,:,:] = upFeat
                return innerFeat, light#light


        class HourglassNet(nn.Module):
            def __init__(self, baseFilter = 16, gray=True):
                super(HourglassNet, self).__init__()

                self.ncLight = 27   # number of channels for input to lighting network
                self.baseFilter = baseFilter

                # number of channles for output of lighting network
                if gray:
                    self.ncOutLight = 9  # gray: channel is 1
                else:
                    self.ncOutLight = 27  # color: channel is 3

                self.ncPre = self.baseFilter  # number of channels for pre-convolution

                # number of channels 
                self.ncHG3 = self.baseFilter
                self.ncHG2 = 2*self.baseFilter
                self.ncHG1 = 4*self.baseFilter
                self.ncHG0 = 8*self.baseFilter + self.ncLight

                self.pre_conv = nn.Conv2d(1, self.ncPre, kernel_size=5, stride=1, padding=2)
                self.pre_bn = nn.BatchNorm2d(self.ncPre)

                self.light = lightingNet(self.ncLight, self.ncOutLight, 128)
                self.HG0 = HourglassBlock(self.ncHG1, self.ncHG0, self.light)
                self.HG1 = HourglassBlock(self.ncHG2, self.ncHG1, self.HG0)
                self.HG2 = HourglassBlock(self.ncHG3, self.ncHG2, self.HG1)
                self.HG3 = HourglassBlock(self.ncPre, self.ncHG3, self.HG2)

                self.conv_1 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=3, stride=1, padding=1)
                self.bn_1 = nn.BatchNorm2d(self.ncPre) 
                self.conv_2 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
                self.bn_2 = nn.BatchNorm2d(self.ncPre) 
                self.conv_3 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
                self.bn_3 = nn.BatchNorm2d(self.ncPre)

                self.output = nn.Conv2d(self.ncPre, 1, kernel_size=1, stride=1, padding=0)

            def forward(self, x, target_light, skip_count):
                feat = self.pre_conv(x)
                
                feat = F.relu(self.pre_bn(feat))
                # get the inner most features
                feat, out_light = self.HG3(feat, target_light, 0, skip_count)
                #return feat, out_light
                
                feat = F.relu(self.bn_1(self.conv_1(feat)))
                feat = F.relu(self.bn_2(self.conv_2(feat)))
                feat = F.relu(self.bn_3(self.conv_3(feat)))
                out_img = self.output(feat)
                out_img = torch.sigmoid(out_img)
                return out_img, out_light   
                
        model = HourglassNet()
        t_dict = torch.load( Path(__file__).parent / 'DeepPortraitRelighting.t7' )
        model.load_state_dict(t_dict)
        model.to( torch_device )
        model.train(False)
        return model
