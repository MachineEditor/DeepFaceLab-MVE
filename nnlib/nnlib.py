import os
import sys
import contextlib
import numpy as np

from utils import std_utils
from .devicelib import devicelib

class nnlib(object):
    device = devicelib #forwards nnlib.devicelib to device in order to use nnlib as standalone lib
    DeviceConfig = devicelib.Config
    active_DeviceConfig = DeviceConfig() #default is one best GPU

    dlib = None
    keras = None
    keras_contrib = None
    tf = None
    tf_sess = None
    
    code_import_tf = None
    code_import_keras = None
    code_import_keras_contrib = None
    code_import_all = None
    
    code_import_dlib = None

    tf_dssim = None
    tf_ssim = None
    tf_resize_like = None
    tf_image_histogram = None
    tf_rgb_to_lab = None
    tf_lab_to_rgb = None
    tf_adain = None
    tf_gaussian_blur = None
    tf_style_loss = None
    
    modelify = None
    ReflectionPadding2D = None
    DSSIMLoss = None
    DSSIMMaskLoss = None
    PixelShuffler = None  
    SubpixelUpscaler = None
    AddUniformNoise = None
    
    ResNet = None
    UNet = None
    UNetTemporalPredictor = None
    NLayerDiscriminator = None
    
    code_import_tf_string = \
"""
tf = nnlib.tf
tf_sess = nnlib.tf_sess

tf_total_variation = tf.image.total_variation
tf_dssim = nnlib.tf_dssim
tf_ssim = nnlib.tf_ssim
tf_resize_like = nnlib.tf_resize_like
tf_image_histogram = nnlib.tf_image_histogram
tf_rgb_to_lab = nnlib.tf_rgb_to_lab
tf_lab_to_rgb = nnlib.tf_lab_to_rgb
tf_adain = nnlib.tf_adain
tf_gaussian_blur = nnlib.tf_gaussian_blur
tf_style_loss = nnlib.tf_style_loss
"""
    code_import_keras_string = \
"""
keras = nnlib.keras
K = keras.backend

Input = keras.layers.Input

Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
SeparableConv2D = keras.layers.SeparableConv2D
MaxPooling2D = keras.layers.MaxPooling2D
BatchNormalization = keras.layers.BatchNormalization

LeakyReLU = keras.layers.LeakyReLU
ReLU = keras.layers.ReLU
tanh = keras.layers.Activation('tanh')
sigmoid = keras.layers.Activation('sigmoid')
Dropout = keras.layers.Dropout

Add = keras.layers.Add
Concatenate = keras.layers.Concatenate

Flatten = keras.layers.Flatten
Reshape = keras.layers.Reshape

ZeroPadding2D = keras.layers.ZeroPadding2D       

RandomNormal = keras.initializers.RandomNormal
Model = keras.models.Model

Adam = keras.optimizers.Adam

modelify = nnlib.modelify
ReflectionPadding2D = nnlib.ReflectionPadding2D
DSSIMLoss = nnlib.DSSIMLoss
DSSIMMaskLoss = nnlib.DSSIMMaskLoss
PixelShuffler = nnlib.PixelShuffler
SubpixelUpscaler = nnlib.SubpixelUpscaler
AddUniformNoise = nnlib.AddUniformNoise
"""
    code_import_keras_contrib_string = \
"""
keras_contrib = nnlib.keras_contrib
GroupNormalization = keras_contrib.layers.GroupNormalization
InstanceNormalization = keras_contrib.layers.InstanceNormalization
Padam = keras_contrib.optimizers.Padam
"""
    code_import_dlib_string = \
"""
dlib = nnlib.dlib
"""    

    code_import_all_string = \
"""
ResNet = nnlib.ResNet
UNet = nnlib.UNet
UNetTemporalPredictor = nnlib.UNetTemporalPredictor
NLayerDiscriminator = nnlib.NLayerDiscriminator
"""
    
            
    @staticmethod
    def import_tf(device_config = None):
        if nnlib.tf is not None:
            return nnlib.code_import_tf

        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
            suppressor = std_utils.suppress_stdout_stderr().__enter__()
        else:
            suppressor = None
            
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        
        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        
        import tensorflow as tf
        nnlib.tf = tf
        
        if device_config is None:
            device_config = nnlib.active_DeviceConfig
        
        tf_ver = [int(x) for x in tf.VERSION.split('.')]
        req_cap = 35
        if tf_ver[0] > 1 or (tf_ver[0] == 1 and tf_ver[1] >= 11):
            req_cap = 37
            
        if not device_config.cpu_only and device_config.gpu_compute_caps[0] < req_cap:
            if suppressor is not None:  
                suppressor.__exit__()
            
            print ("%s does not meet minimum required compute capability: %d.%d. Falling back to CPU mode." % ( device_config.gpu_names[0], req_cap // 10, req_cap % 10 ) )
            device_config = nnlib.DeviceConfig(cpu_only=True)
            
            if suppressor is not None:  
                suppressor.__enter__()

        nnlib.active_DeviceConfig = device_config
        
        if device_config.cpu_only:
            config = tf.ConfigProto( device_count = {'GPU': 0} )
        else:     
            config = tf.ConfigProto()
            visible_device_list = ''
            for idx in device_config.gpu_idxs:
                visible_device_list += str(idx) + ','
            config.gpu_options.visible_device_list=visible_device_list[:-1]
            
        config.gpu_options.force_gpu_compatible = True            
        config.gpu_options.allow_growth = device_config.allow_growth
        
        nnlib.tf_sess = tf.Session(config=config)
            
        if suppressor is not None:  
            suppressor.__exit__()

        nnlib.__initialize_tf_functions()
        nnlib.code_import_tf = compile (nnlib.code_import_tf_string,'','exec')
        return nnlib.code_import_tf
        
    @staticmethod
    def __initialize_tf_functions():
        tf = nnlib.tf

        def tf_dssim_(max_value=1.0):
            def func(t1,t2):
                return (1.0 - tf.image.ssim (t1, t2, max_value)) / 2.0
            return func
        nnlib.tf_dssim = tf_dssim_
         
        def tf_ssim_(max_value=1.0):            
            def func(t1,t2):
                return tf.image.ssim (t1, t2, max_value)
            return func
        nnlib.tf_ssim = tf_ssim_
        
        def tf_resize_like_(ref_tensor):
            def func(input_tensor):
                H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
                return tf.image.resize_bilinear(input_tensor, [H.value, W.value])
            return func
        nnlib.tf_resize_like = tf_resize_like_

        def tf_rgb_to_lab():
            def func(rgb_input):
                with tf.name_scope("rgb_to_lab"):
                    srgb_pixels = tf.reshape(rgb_input, [-1, 3])

                    with tf.name_scope("srgb_to_xyz"):
                        linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
                        exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
                        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
                        rgb_to_xyz = tf.constant([
                            #    X        Y          Z
                            [0.412453, 0.212671, 0.019334], # R
                            [0.357580, 0.715160, 0.119193], # G
                            [0.180423, 0.072169, 0.950227], # B
                        ])
                        xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

                    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
                    with tf.name_scope("xyz_to_cielab"):
                        # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

                        # normalize for D65 white point
                        xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

                        epsilon = 6/29
                        linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
                        exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
                        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

                        # convert to lab
                        fxfyfz_to_lab = tf.constant([
                            #  l       a       b
                            [  0.0,  500.0,    0.0], # fx
                            [116.0, -500.0,  200.0], # fy
                            [  0.0,    0.0, -200.0], # fz
                        ])
                        lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
                    return tf.reshape(lab_pixels, tf.shape(rgb_input))
            return func
        nnlib.tf_rgb_to_lab = tf_rgb_to_lab
        
        def tf_lab_to_rgb():
            def func(lab):
                with tf.name_scope("lab_to_rgb"):
                    lab_pixels = tf.reshape(lab, [-1, 3])

                    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
                    with tf.name_scope("cielab_to_xyz"):
                        # convert to fxfyfz
                        lab_to_fxfyfz = tf.constant([
                            #   fx      fy        fz
                            [1/116.0, 1/116.0,  1/116.0], # l
                            [1/500.0,     0.0,      0.0], # a
                            [    0.0,     0.0, -1/200.0], # b
                        ])
                        fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

                        # convert to xyz
                        epsilon = 6/29
                        linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
                        exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
                        xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

                        # denormalize for D65 white point
                        xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

                    with tf.name_scope("xyz_to_srgb"):
                        xyz_to_rgb = tf.constant([
                            #     r           g          b
                            [ 3.2404542, -0.9692660,  0.0556434], # x
                            [-1.5371385,  1.8760108, -0.2040259], # y
                            [-0.4985314,  0.0415560,  1.0572252], # z
                        ])
                        rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
                        # avoid a slightly negative number messing up the conversion
                        rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
                        linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
                        exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
                        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

                    return tf.reshape(srgb_pixels, tf.shape(lab))
            return func
        nnlib.tf_lab_to_rgb = tf_lab_to_rgb

        def tf_image_histogram():
            def func(input):
                x = input
                x += 1 / 255.0
                
                output = []
                for i in range(256, 0, -1):
                    v = i / 255.0
                    y = (x - v) * 1000
                    
                    y = tf.clip_by_value (y, -1.0, 0.0) + 1

                    output.append ( tf.reduce_sum (y) )
                    x -= y*v

                return tf.stack ( output[::-1] )
            return func
        nnlib.tf_image_histogram = tf_image_histogram
     
        def tf_adain(epsilon=1e-5):
            def func(content, style):
                axes = [1,2]
                c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
                s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
                c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)
                return s_std * (content - c_mean) / c_std + s_mean
            return func
        nnlib.tf_adain = tf_adain
        
        def tf_gaussian_blur(radius=2.0):
            def gaussian_kernel(size,mean,std):
                d = tf.distributions.Normal( float(mean), float(std) )

                vals = d.prob(tf.range(start = -int(size), limit = int(size) + 1, dtype = tf.float32))

                gauss_kernel = tf.einsum('i,j->ij',
                                              vals,
                                              vals)

                return gauss_kernel / tf.reduce_sum(gauss_kernel)

            gauss_kernel = gaussian_kernel(radius, 1.0, radius )
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            
            def func(input):
                return tf.nn.conv2d(input, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
            return func
        nnlib.tf_gaussian_blur = tf_gaussian_blur

        def tf_style_loss(gaussian_blur_radius=0.0, loss_weight=1.0, batch_normalize=False, epsilon=1e-5):
            def sl(content, style):
                axes = [1,2]
                c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
                s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
                c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

                mean_loss = tf.reduce_sum(tf.squared_difference(c_mean, s_mean))
                std_loss = tf.reduce_sum(tf.squared_difference(c_std, s_std))

                if batch_normalize:
                    #normalize w.r.t batch size
                    n = tf.cast(tf.shape(content)[0], dtype=tf.float32)
                    mean_loss /= n
                    std_loss /= n
                
                return (mean_loss + std_loss) * loss_weight
                
            def func(target, style):
                target_nc = target.get_shape().as_list()[-1]
                style_nc = style.get_shape().as_list()[-1]
                if target_nc != style_nc:
                    raise Exception("target_nc != style_nc")
                   
                targets = tf.split(target, target_nc, -1)
                styles = tf.split(style, style_nc, -1)
                
                style_loss = []
                for i in range(len(targets)):
                    if gaussian_blur_radius > 0.0:
                        style_loss += [ sl( tf_gaussian_blur(gaussian_blur_radius)(targets[i]), 
                                            tf_gaussian_blur(gaussian_blur_radius)(styles[i]))  ]
                    else:
                        style_loss += [ sl( targets[i], 
                                            styles[i])  ]
                return np.sum ( style_loss )  
            return func
            
        nnlib.tf_style_loss = tf_style_loss

    @staticmethod
    def import_keras(device_config = None):
        if nnlib.keras is not None:
            return nnlib.code_import_keras

        nnlib.import_tf(device_config)
        device_config = nnlib.active_DeviceConfig
        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
            suppressor = std_utils.suppress_stdout_stderr().__enter__()
            
        import keras as keras_
        nnlib.keras = keras_
        
        if device_config.use_fp16:
            nnlib.keras.backend.set_floatx('float16')
        
        nnlib.keras.backend.set_session(nnlib.tf_sess)
        nnlib.keras.backend.set_image_data_format('channels_last')
        
        if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':        
            suppressor.__exit__()

        nnlib.__initialize_keras_functions()  
        nnlib.code_import_keras = compile (nnlib.code_import_keras_string,'','exec')

        
    @staticmethod
    def __initialize_keras_functions():
        tf = nnlib.tf
        keras = nnlib.keras
        K = keras.backend

        def modelify(model_functor):
            def func(tensor):
                return keras.models.Model (tensor, model_functor(tensor))
            return func
        
        nnlib.modelify = modelify
        
        class ReflectionPadding2D(keras.layers.Layer):
            def __init__(self, padding=(1, 1), **kwargs):
                self.padding = tuple(padding)
                self.input_spec = [keras.layers.InputSpec(ndim=4)]
                super(ReflectionPadding2D, self).__init__(**kwargs)

            def compute_output_shape(self, s):
                """ If you are using "channels_last" configuration"""
                return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

            def call(self, x, mask=None):
                w_pad,h_pad = self.padding
                return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
        nnlib.ReflectionPadding2D = ReflectionPadding2D

        class DSSIMLoss(object):
            def __init__(self, is_tanh=False):
                self.is_tanh = is_tanh
                
            def __call__(self,y_true, y_pred):
                if not self.is_tanh:            
                    return (1.0 - tf.image.ssim (y_true, y_pred, 1.0)) / 2.0
                else:
                    return (1.0 - tf.image.ssim ((y_true/2+0.5), (y_pred/2+0.5), 1.0)) / 2.0
        nnlib.DSSIMLoss = DSSIMLoss

        class DSSIMLoss(object):
            def __init__(self, is_tanh=False):
                self.is_tanh = is_tanh
                
            def __call__(self,y_true, y_pred):

                if not self.is_tanh:            
                    loss = (1.0 - tf.image.ssim (y_true, y_pred, 1.0)) / 2.0
                else:
                    loss = (1.0 - tf.image.ssim ( (y_true/2+0.5), (y_pred/2+0.5), 1.0)) / 2.0

                return loss
        nnlib.DSSIMLoss = DSSIMLoss
        
        class DSSIMMaskLoss(object):
            def __init__(self, mask_list, is_tanh=False):
                self.mask_list = mask_list
                self.is_tanh = is_tanh
                
            def __call__(self,y_true, y_pred):
                total_loss = None
                for mask in self.mask_list:
                
                    if not self.is_tanh:            
                        loss = (1.0 - (tf.image.ssim (y_true*mask, y_pred*mask, 1.0))) / 2.0
                    else:
                        loss = (1.0 - tf.image.ssim ( (y_true/2+0.5)*(mask/2+0.5), (y_pred/2+0.5)*(mask/2+0.5), 1.0)) / 2.0
                    
                    loss = K.cast (loss, K.floatx())
                    
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss += loss
                        
                return total_loss
        nnlib.DSSIMMaskLoss = DSSIMMaskLoss
        
        class PixelShuffler(keras.layers.Layer):
            def __init__(self, size=(2, 2), data_format=None, **kwargs):
                super(PixelShuffler, self).__init__(**kwargs)
                self.data_format = keras.backend.common.normalize_data_format(data_format)
                self.size = keras.utils.conv_utils.normalize_tuple(size, 2, 'size')

            def call(self, inputs):
                input_shape = keras.backend.int_shape(inputs)
                if len(input_shape) != 4:
                    raise ValueError('Inputs should have rank ' +
                                     str(4) +
                                     '; Received input shape:', str(input_shape))

                if self.data_format == 'channels_first':
                    return tf.depth_to_space(inputs, self.size[0], 'NCHW')

                elif self.data_format == 'channels_last':
                    return tf.depth_to_space(inputs, self.size[0], 'NHWC')

            def compute_output_shape(self, input_shape):
                if len(input_shape) != 4:
                    raise ValueError('Inputs should have rank ' +
                                     str(4) +
                                     '; Received input shape:', str(input_shape))

                if self.data_format == 'channels_first':
                    height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
                    width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
                    channels = input_shape[1] // self.size[0] // self.size[1]

                    if channels * self.size[0] * self.size[1] != input_shape[1]:
                        raise ValueError('channels of input and size are incompatible')

                    return (input_shape[0],
                            channels,
                            height,
                            width)

                elif self.data_format == 'channels_last':
                    height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
                    width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
                    channels = input_shape[3] // self.size[0] // self.size[1]

                    if channels * self.size[0] * self.size[1] != input_shape[3]:
                        raise ValueError('channels of input and size are incompatible')

                    return (input_shape[0],
                            height,
                            width,
                            channels)

            def get_config(self):
                config = {'size': self.size,
                          'data_format': self.data_format}
                base_config = super(PixelShuffler, self).get_config()

                return dict(list(base_config.items()) + list(config.items()))

        nnlib.PixelShuffler = PixelShuffler
        nnlib.SubpixelUpscaler = PixelShuffler
        
        class AddUniformNoise(keras.layers.Layer):
            def __init__(self, power=1.0, minval=-1.0, maxval=1.0, **kwargs):
                super(AddUniformNoise, self).__init__(**kwargs)
                self.power = power
                self.supports_masking = True
                self.minval = minval
                self.maxval = maxval

            def call(self, inputs, training=None):
                def noised():
                    return inputs + self.power*K.random_uniform(shape=K.shape(inputs), minval=self.minval, maxval=self.maxval)
                return K.in_train_phase(noised, inputs, training=training)

            def get_config(self):
                config = {'power': self.power, 'minval': self.minval, 'maxval': self.maxval}
                base_config = super(AddUniformNoise, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        nnlib.AddUniformNoise = AddUniformNoise       
                
    @staticmethod
    def import_keras_contrib(device_config = None):
        if nnlib.keras_contrib is not None:
            return nnlib.code_import_keras_contrib
        
        import keras_contrib as keras_contrib_    
        nnlib.keras_contrib = keras_contrib_
        nnlib.__initialize_keras_contrib_functions()    
        nnlib.code_import_keras_contrib = compile (nnlib.code_import_keras_contrib_string,'','exec')
    
    @staticmethod
    def __initialize_keras_contrib_functions():
        pass
        
    @staticmethod
    def import_dlib( device_config = None):
        if nnlib.dlib is not None:
            return nnlib.code_import_dlib

        import dlib as dlib_
        nnlib.dlib = dlib_
        if not device_config.cpu_only and len(device_config.gpu_idxs) > 0:
            nnlib.dlib.cuda.set_device(device_config.gpu_idxs[0])
            
        
        nnlib.code_import_dlib = compile (nnlib.code_import_dlib_string,'','exec')
    
    @staticmethod
    def import_all(device_config = None):
        if nnlib.code_import_all is None:
            nnlib.import_tf(device_config)        
            nnlib.import_keras(device_config)
            nnlib.import_keras_contrib(device_config)        
            nnlib.code_import_all = compile (nnlib.code_import_tf_string + '\n' 
                                            + nnlib.code_import_keras_string + '\n'
                                            + nnlib.code_import_keras_contrib_string 
                                            + nnlib.code_import_all_string,'','exec')        
            nnlib.__initialize_all_functions()
        
        return nnlib.code_import_all
    
    @staticmethod
    def __initialize_all_functions():
        def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
            exec (nnlib.import_all(), locals(), globals())

            if not use_batch_norm:
                use_bias = True
                def XNormalization(x):
                    return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)#GroupNormalization (axis=3, groups=K.int_shape (x)[3] // 4, gamma_initializer=RandomNormal(1., 0.02))(x)
            else:
                use_bias = False
                def XNormalization(x):
                    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
                    
            def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )
        
            def Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
            
            def func(input):

               
                def ResnetBlock(dim):    
                    def func(input):
                        x = input
                        
                        x = ReflectionPadding2D((1,1))(x)
                        x = Conv2D(dim, 3, 1, padding='valid')(x)
                        x = XNormalization(x)
                        x = ReLU()(x)
                        
                        if use_dropout:
                            x = Dropout(0.5)(x)
                            
                        x = ReflectionPadding2D((1,1))(x)
                        x = Conv2D(dim, 3, 1, padding='valid')(x)
                        x = XNormalization(x)
                        x = ReLU()(x)       
                        return Add()([x,input])            
                    return func

                x = input
                
                x = ReflectionPadding2D((3,3))(x)
                x = Conv2D(ngf, 7, 1, 'valid')(x)
                
                x = ReLU()(XNormalization(Conv2D(ngf*2, 4, 2, 'same')(x)))
                x = ReLU()(XNormalization(Conv2D(ngf*4, 4, 2, 'same')(x)))
                
                for i in range(n_blocks):
                    x = ResnetBlock(ngf*4)(x)
                    
                x = ReLU()(XNormalization(PixelShuffler()(Conv2D(ngf*2 *4, 3, 1, 'same')(x))))
                x = ReLU()(XNormalization(PixelShuffler()(Conv2D(ngf   *4, 3, 1, 'same')(x))))
                
                x = ReflectionPadding2D((3,3))(x)
                x = Conv2D(output_nc, 7, 1, 'valid')(x)
                x = tanh(x)
                
                return x
                
            return func  
            
        nnlib.ResNet = ResNet
             
        # Defines the Unet generator.
        # |num_downs|: number of downsamplings in UNet. For example,
        # if |num_downs| == 7, image of size 128x128 will become of size 1x1
        # at the bottleneck
        def UNet(output_nc, use_batch_norm, num_downs, ngf=64, use_dropout=False):
            exec (nnlib.import_all(), locals(), globals())

            if not use_batch_norm:
                use_bias = True
                def XNormalization(x):
                    return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)#GroupNormalization (axis=3, groups=K.int_shape (x)[3] // 4, gamma_initializer=RandomNormal(1., 0.02))(x)
            else:
                use_bias = False
                def XNormalization(x):
                    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
                    
            def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

            def Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
                
            def UNetSkipConnection(outer_nc, inner_nc, sub_model=None, outermost=False, innermost=False, use_dropout=False):       
                def func(inp):
                    x = inp
                    
                    x = Conv2D(inner_nc, 4, 2, 'valid')(ReflectionPadding2D( (1,1) )(x))
                    x = XNormalization(x)
                    x = ReLU()(x)
                        
                    if not innermost:
                        x = sub_model(x)
                        
                    if not outermost:
                        x = Conv2DTranspose(outer_nc, 3, 2, 'same')(x)
                        x = XNormalization(x)
                        x = ReLU()(x)
                        
                        if not innermost:
                            if use_dropout:
                                x = Dropout(0.5)(x)
                            
                        x = Concatenate(axis=3)([inp, x])
                    else:
                        x = Conv2DTranspose(outer_nc, 3, 2, 'same')(x)
                        x = tanh(x)   
                        

                    return x           
                    
                return func
                    
            def func(input):                    

                unet_block = UNetSkipConnection(ngf * 8, ngf * 8, sub_model=None, innermost=True)

                #for i in range(num_downs - 5):
                #    unet_block = UNetSkipConnection(ngf * 8, ngf * 8, sub_model=unet_block, use_dropout=use_dropout)
                
                unet_block = UNetSkipConnection(ngf * 4  , ngf * 8, sub_model=unet_block)
                unet_block = UNetSkipConnection(ngf * 2  , ngf * 4, sub_model=unet_block)
                unet_block = UNetSkipConnection(ngf      , ngf * 2, sub_model=unet_block)
                unet_block = UNetSkipConnection(output_nc, ngf    , sub_model=unet_block, outermost=True)
                
                return unet_block(input)
            return func
        nnlib.UNet = UNet
        
        #predicts based on two past_image_tensors
        def UNetTemporalPredictor(output_nc, use_batch_norm, num_downs, ngf=64, use_dropout=False):
            exec (nnlib.import_all(), locals(), globals())
            def func(inputs):
                past_2_image_tensor, past_1_image_tensor = inputs
                    
                x = Concatenate(axis=3)([ past_2_image_tensor, past_1_image_tensor ])
                x = UNet(3, use_batch_norm, num_downs=num_downs, ngf=ngf, use_dropout=use_dropout) (x)

                return x
                    
            return func                
        nnlib.UNetTemporalPredictor = UNetTemporalPredictor
        
        def NLayerDiscriminator(use_batch_norm, ndf=64, n_layers=3):
            exec (nnlib.import_all(), locals(), globals())
            
            if not use_batch_norm:
                use_bias = True
                def XNormalization(x):
                    return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)#GroupNormalization (axis=3, groups=K.int_shape (x)[3] // 4, gamma_initializer=RandomNormal(1., 0.02))(x)
            else:
                use_bias = False
                def XNormalization(x):
                    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

            def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
                return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

            def func(input):
                x = input
                
                x = ZeroPadding2D((1,1))(x)
                x = Conv2D( ndf, 4, 2, 'valid')(x)
                x = LeakyReLU(0.2)(x)
                    
                for i in range(1, n_layers):        
                    x = ZeroPadding2D((1,1))(x)
                    x = Conv2D( ndf * min(2 ** i, 8), 4, 2, 'valid')(x)
                    x = XNormalization(x)
                    x = LeakyReLU(0.2)(x)
                    
                x = ZeroPadding2D((1,1))(x)
                x = Conv2D( ndf * min(2 ** n_layers, 8), 4, 1, 'valid')(x)
                x = XNormalization(x)    
                x = LeakyReLU(0.2)(x)
                
                x = ZeroPadding2D((1,1))(x)
                return Conv2D( 1, 4, 1, 'valid')(x)
            return func
        nnlib.NLayerDiscriminator = NLayerDiscriminator
        
    @staticmethod
    def finalize_all():
        if nnlib.keras_contrib is not None:
            nnlib.keras_contrib = None
        
        if nnlib.keras is not None:
            nnlib.keras.backend.clear_session()
            nnlib.keras = None

        if nnlib.tf is not None:
            nnlib.tf_sess.close()
            nnlib.tf_sess = None
            nnlib.tf = None


