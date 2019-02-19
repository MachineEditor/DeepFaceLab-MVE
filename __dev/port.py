#import FaceLandmarksExtractor


import numpy as np
import dlib
import torch
import keras
from keras import backend as K
from keras import layers as KL
import math
import os
import time
import code

class TorchBatchNorm2D(keras.engine.topology.Layer):
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
        super(TorchBatchNorm2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape, name='gamma', initializer='ones', regularizer=None, constraint=None)
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros', regularizer=None, constraint=None)
        self.moving_mean = self.add_weight(shape=shape, name='moving_mean', initializer='zeros', trainable=False)            
        self.moving_variance = self.add_weight(shape=shape, name='moving_variance', initializer='ones', trainable=False)            
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        
        broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
        broadcast_moving_variance = K.reshape(self.moving_variance, broadcast_shape)
        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)        
        invstd = K.ones (shape=broadcast_shape, dtype='float32') / K.sqrt(broadcast_moving_variance + K.constant(self.epsilon, dtype='float32'))
        
        return (inputs - broadcast_moving_mean) * invstd * broadcast_gamma + broadcast_beta
       
    def get_config(self):
        config = { 'axis': self.axis, 'momentum': self.momentum, 'epsilon': self.epsilon }
        base_config = super(TorchBatchNorm2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

       
def t2kw_conv2d (src):
    if src.bias is not None:
        return [ np.moveaxis(src.weight.data.cpu().numpy(), [0,1,2,3], [3,2,0,1]), src.bias.data.cpu().numpy() ]
    else:
        return [ np.moveaxis(src.weight.data.cpu().numpy(), [0,1,2,3], [3,2,0,1])]
        
    
def t2kw_bn2d(src):
    return [ src.weight.data.cpu().numpy(), src.bias.data.cpu().numpy(), src.running_mean.cpu().numpy(), src.running_var.cpu().numpy() ]


    
import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,enable_cuda=False,enable_cudnn=False,use_cnn_face_detector=True).face_alignemnt_net
fa.eval()


def KerasConvBlock(in_planes, out_planes, input, srctorch):
    out1 = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.bn1) )(input)
    out1 = KL.Activation( keras.backend.relu ) (out1)
    out1 = KL.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(out1)
    out1 = KL.convolutional.Conv2D( int(out_planes/2), kernel_size=3, strides=1, data_format='channels_first', padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.conv1) ) (out1)
     
    out2 = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.bn2) )(out1)
    out2 = KL.Activation( keras.backend.relu ) (out2)
    out2 = KL.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(out2)
    out2 = KL.convolutional.Conv2D( int(out_planes/4), kernel_size=3, strides=1, data_format='channels_first', padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.conv2) ) (out2)
    
    out3 = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.bn3) )(out2)
    out3 = KL.Activation( keras.backend.relu ) (out3)
    out3 = KL.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(out3)
    out3 = KL.convolutional.Conv2D( int(out_planes/4), kernel_size=3, strides=1, data_format='channels_first', padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.conv3) ) (out3)
     
    out3 = KL.Concatenate(axis=1)([out1, out2, out3])
    
    if in_planes != out_planes:
        downsample = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.downsample[0]) )(input)
        downsample = KL.Activation( keras.backend.relu ) (downsample)
        downsample = KL.convolutional.Conv2D( out_planes, kernel_size=1, strides=1, data_format='channels_first', padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.downsample[2]) ) (downsample)
        out3 = KL.add ( [out3, downsample] )
    else:
        out3 = KL.add ( [out3, input] )
    

    return out3
    
def KerasHourGlass (depth, input, srctorch):

    up1 = KerasConvBlock(256, 256, input, srctorch._modules['b1_%d' % (depth)])
    
    low1 = KL.AveragePooling2D (pool_size=2, strides=2, data_format='channels_first', padding='valid' )(input)
    low1 = KerasConvBlock (256, 256, low1, srctorch._modules['b2_%d' % (depth)])
    
    if depth > 1:
        low2 = KerasHourGlass (depth-1, low1, srctorch)
    else:
        low2 = KerasConvBlock(256, 256, low1, srctorch._modules['b2_plus_%d' % (depth)])
    
    low3 = KerasConvBlock(256, 256, low2, srctorch._modules['b3_%d' % (depth)])
    
    up2 = KL.UpSampling2D(size=2, data_format='channels_first') (low3)
    return KL.add ( [up1, up2] )
    
model_path = os.path.join( os.path.dirname(__file__) , "2DFAN-4.h5" )
if os.path.exists (model_path):    
    t = time.time()
    model = keras.models.load_model (model_path, custom_objects={'TorchBatchNorm2D': TorchBatchNorm2D} ) 
    print ('load takes = %f' %( time.time() - t ) )
else:
    _input = keras.layers.Input ( shape=(3, 256,256) )
    x = KL.ZeroPadding2D(padding=(3, 3), data_format='channels_first')(_input)
    x = KL.convolutional.Conv2D( 64, kernel_size=7, strides=2, data_format='channels_first', padding='valid', weights=t2kw_conv2d(fa.conv1) ) (x)
    
    x = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(fa.bn1) )(x)
    x = KL.Activation( keras.backend.relu ) (x)
    
    x = KerasConvBlock (64, 128, x, fa.conv2)
    x = KL.AveragePooling2D (pool_size=2, strides=2, data_format='channels_first', padding='valid' ) (x)
    x = KerasConvBlock (128, 128, x, fa.conv3)
    x = KerasConvBlock (128, 256, x, fa.conv4)
    
    outputs = []
    previous = x
    for i in range(4):
        ll = KerasHourGlass (4, previous, fa._modules['m%d' % (i) ])
        ll = KerasConvBlock (256,256, ll, fa._modules['top_m_%d' % (i)])
        
        ll = KL.convolutional.Conv2D(256, kernel_size=1, strides=1, data_format='channels_first', padding='valid', weights=t2kw_conv2d( fa._modules['conv_last%d' % (i)] ) ) (ll)
        ll = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d( fa._modules['bn_end%d' % (i)] ) )(ll)
        ll = KL.Activation( keras.backend.relu ) (ll)
        
        tmp_out = KL.convolutional.Conv2D(68, kernel_size=1, strides=1, data_format='channels_first', padding='valid', weights=t2kw_conv2d( fa._modules['l%d' % (i)] ) ) (ll)
        outputs.append(tmp_out)
        if i < 4 - 1:
            ll = KL.convolutional.Conv2D(256, kernel_size=1, strides=1, data_format='channels_first', padding='valid', weights=t2kw_conv2d( fa._modules['bl%d' % (i)] ) ) (ll)
            previous = KL.add ( [previous, ll, KL.convolutional.Conv2D(256, kernel_size=1, strides=1, data_format='channels_first', padding='valid', weights=t2kw_conv2d( fa._modules['al%d' % (i)] ) ) (tmp_out) ] )
            
    model = keras.models.Model (_input, outputs)
    model.compile ( loss='mse', optimizer='adam' )
    model.save (model_path)
    model.save_weights ( os.path.join( os.path.dirname(__file__) , 'weights.h5') )
    
def transform(point, center, scale, resolution, invert=False):
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()
    
def get_preds_fromhm(hm, center=None, scale=None):
    max, idx = torch.max(  hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)
    
    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)
                    
    return preds, preds_orig


def get_preds_fromhm2(a, center=None, scale=None):
    b = a.reshape ( (a.shape[0], a.shape[1]*a.shape[2]) )    
    c = b.argmax(1).reshape ( (a.shape[0], 1) ).repeat(2, axis=1).astype(np.float)
    c[:,0] %= a.shape[2]    
    c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a.shape[2]), 0, c[:,1] )

    for i in range(a.shape[0]):
        pX, pY = int(c[i,0]), int(c[i,1])
        if pX > 0 and pX < 63 and pY > 0 and pY < 63:
            diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
            c[i] += np.sign(diff)*0.25
   
    c += 0.5
    result = np.empty ( (a.shape[0],2), dtype=np.int )
    if center is not None and scale is not None:
        for i in range(a.shape[0]):
            pt = np.array ( [c[i][0], c[i][1], 1.0] )            
            h = 200.0 * scale
            m = np.eye(3)
            m[0,0] = a.shape[2] / h
            m[1,1] = a.shape[2] / h
            m[0,2] = a.shape[2] * ( -center[0] / h + 0.5 )
            m[1,2] = a.shape[2] * ( -center[1] / h + 0.5 )
            m = np.linalg.inv(m)
            result[i] = np.matmul (m, pt)[0:2].astype( np.int )
    return result
    

    
rnd_data = np.random.rand (3, 256,256).astype(np.float32)
#rnd_data = np.random.random_integers (2, size=(3, 256,256)).astype(np.float32)
#rnd_data = np.array ( [[[1]*256]*256]*3 , dtype=np.float32 )
input_data = np.array ([rnd_data])

fa_out_tensor = fa( torch.autograd.Variable( torch.from_numpy(input_data), volatile=True) )[-1].data.cpu()
fa_out = fa_out_tensor.numpy()

t = time.time()
m_out = model.predict ( input_data )[-1]
print ('predict takes = %f' %( time.time() - t ) )
t = time.time()

#fa_base_out = fa_base(torch.autograd.Variable( torch.from_numpy(input_data), volatile=True))[0].data.cpu().numpy()

print ( 'shapes = %s , %s , equal == %s ' % (fa_out.shape, m_out.shape, (fa_out.shape == m_out.shape) ) )
print ( 'allclose == %s' %  ( np.allclose(fa_out, m_out) ) )
print ( 'total abs diff outputs = %f' % ( np.sum ( np.abs(np.ndarray.flatten(fa_out-m_out))) )) 

###
d = dlib.rectangle(156,364,424,765)

center = torch.FloatTensor(
                    [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                     (d.bottom() - d.top()) / 2.0])
center[1] = center[1] - (d.bottom() - d.top()) * 0.12
scale = (d.right() - d.left() + d.bottom() - d.top()) / 195.0
pts, pts_img = get_preds_fromhm (fa_out_tensor, center, scale)
pts_img = pts_img.view(68, 2).numpy()

###

m_pts_img = get_preds_fromhm2 (m_out[0], center, scale)

print ('pts1 == pts2 == %s' % ( np.array_equal(pts_img, m_pts_img) ) )

code.interact(local=dict(globals(), **locals()))

#print ( np.array_equal (fa_out, m_out) ) #>>> False
#code.interact(local=dict(globals(), **locals()))

#code.interact(local=locals())

#code.interact(local=locals())

###
#fa.conv1.weight = torch.nn.Parameter( torch.from_numpy ( np.array( [[[[1.0]*7]*7]*3]*64, dtype=np.float32) ) )
#fa.conv1.bias = torch.nn.Parameter( torch.from_numpy ( np.array( [1.0]*64, dtype=np.float32 ) ) )
#model.layers[2].set_weights( [ np.array( [[[[1.0]*64]*3]*7]*7, dtype=np.float32), np.array( [1.0]*64, dtype=np.float32 ) ] )

#b = np.array( [1.0]*64, dtype=np.float32 )
#b = np.random.rand (64).astype(np.float32)
#w = np.array( [[[[1.0]*7]*7]*3]*64, dtype=np.float32)
#w = np.random.rand (64, 3, 7, 7).astype(np.float32)
#s = w #fa_base.conv1.weight.data.cpu().numpy() #64x3x7x7
#d = np.moveaxis(s, [0,1,2,3], [3,2,0,1] )                
                

#fa.conv1.weight = torch.nn.Parameter( torch.from_numpy ( w ) )
#fa.conv1.bias = torch.nn.Parameter( torch.from_numpy ( b ) )
#model.layers[2].set_weights( [np.transpose(w), b] )
#model.layers[2].set_weights( [d, b] )
'''
for i in range(0,64):
    for j in range(0,128):
        b = np.array_equal (fa_out[i,j], m_out[i,j])
        if b == False:
            print ( '%d %d == False' %(i,j) ) #>>> False
'''      

    
'''
input = -2.7966828
gamma = 0.7640695571899414
beta = 0.22801123559474945
moving_mean = 0.12693816423416138
moving_variance = 0.10409101098775864
epsilon = 0.0 #0.00001

print ( gamma * (input - moving_mean) / math.sqrt(moving_variance + epsilon) + beta )
print ( (input - moving_mean) * (1.0 / math.sqrt(moving_variance) + epsilon)*gamma + beta   )
'''
#code.interact(local=dict(globals(), **locals()))
'''
conv_64_128 = x
conv_64_128 = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(fa.conv2.bn1) )(conv_64_128)
conv_64_128 = KL.Activation( keras.backend.relu ) (conv_64_128)
conv_64_128 = KL.ZeroPadding2D(padding=(1, 1), data_format='channels_first')(conv_64_128)
conv_64_128 = KL.convolutional.Conv2D( 64, kernel_size=3, strides=1, data_format='channels_first', padding='valid', use_bias = False, weights=t2kw_conv2d(fa.conv2.conv1) ) (conv_64_128)
conv_64_128 = TorchBatchNorm2D(axis=1,  momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(fa.conv2.bn2) )(conv_64_128)
conv_64_128 = KL.Activation( keras.backend.relu ) (conv_64_128)
'''
#
#
#keras result = gamma * (input - moving_mean) / sqrt(moving_variance + epsilon) + beta
#
# (input - mean / scale_factor) / sqrt(var / scale_factor + eps)
#
#input = -3.0322433
#
#gamma = 0.1859646
#beta = -0.17041835
#moving_mean = -3.0345056
#moving_variance = 8.773307
#epsilon = 0.00001
#
#result = - 0.17027631
#
# fa result = 1.930317