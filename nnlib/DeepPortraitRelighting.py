from pathlib import Path
import numpy as np
import cv2

class DeepPortraitRelighting(object):
    
    def __init__(self):
        from nnlib import nnlib
        nnlib.import_torch()
        
        self.torch = nnlib.torch
        self.torch_device = nnlib.torch_device
        
        self.model = DeepPortraitRelighting.build_model(self.torch, self.torch_device)
        
        self.shs = [
                    [1.084125496282453138e+00,-4.642676300617166185e-01,2.837846795150648915e-02,6.765292733937575687e-01,-3.594067725393816914e-01,4.790996460111427574e-02,-2.280054643781863066e-01,-8.125983081159608712e-02,2.881082012687687932e-01],
                    [1.084125496282453138e+00,-4.642676300617170626e-01,5.466255701105990905e-01,3.996219229512094628e-01,-2.615439760463462715e-01,-2.511241554473071513e-01,6.495694866016435420e-02,3.510322039081858470e-01,1.189662732386344152e-01],
                    [1.084125496282453138e+00,-4.642676300617179508e-01,6.532524688468428486e-01,-1.782088862752457814e-01,3.326676893441832261e-02,-3.610566644446819295e-01,3.647561777790956361e-01,-7.496419691318900735e-02,-5.412289239602386531e-02],
                    [1.084125496282453138e+00,-4.642676300617186724e-01,2.679669346194941126e-01,-6.218447693376460972e-01,3.030269583891490037e-01,-1.991061409014726058e-01,-6.162944418511027977e-02,-3.176699976873690878e-01,1.920509612235956343e-01],
                    [1.084125496282453138e+00,-4.642676300617186724e-01,-3.191031669056417219e-01,-5.972188577671910803e-01,3.446016675533919993e-01,1.127753677656503223e-01,-1.716692196540034188e-01,2.163406460637767315e-01,2.555824552121269688e-01],
                    [1.084125496282453138e+00,-4.642676300617178398e-01,-6.658820752324799974e-01,-1.228749652534838893e-01,1.266842924569576145e-01,3.397347243069742673e-01,3.036887095295650041e-01,2.213893524577207617e-01,-1.886557316342868038e-02],
                    [1.084125496282453138e+00,-4.642676300617169516e-01,-5.112381993903207800e-01,4.439962822886048266e-01,-1.866289387481862572e-01,3.108669041197227867e-01,2.021743042675238355e-01,-3.148681770175290051e-01,3.974379604123656762e-02]
                   ]
        
    #n = [0..8]
    def relight(self, img, n, lighten=False):
        torch = self.torch
        
        sh = (np.array (self.shs[np.clip(n, 0,8)]).reshape( (1,9,1,1) )*0.7).astype(np.float32)        
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
            outputImg = inputL* outputImg        
        else:
            outputImg = outputImg*255.0
        outputImg = np.clip(outputImg, 0,255).astype(np.uint8)   

        Lab[:,:,0] = outputImg
        result = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        result = cv2.resize(result, (col, row))
        return result
        
    def relight_all(self, img, lighten=False):
        return [ self.relight(img, n, lighten=lighten) for n in range( len(self.shs) ) ]
        
    def relight_random(self, img, lighten=False):       
        return [ self.relight(img, np.random.randint(len(self.shs)), lighten=lighten ) ]
        
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