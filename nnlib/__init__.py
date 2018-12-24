def tf_image_histogram (tf, input):
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
    
def tf_dssim(tf, t1, t2):
    return (1.0 - tf.image.ssim (t1, t2, 1.0)) / 2.0

def tf_ssim(tf, t1, t2):
    return tf.image.ssim (t1, t2, 1.0)
    
def DSSIMMaskLossClass(tf):
    class DSSIMMaskLoss(object):
        def __init__(self, mask_list, is_tanh=False):
            self.mask_list = mask_list
            self.is_tanh = is_tanh
            
        def __call__(self,y_true, y_pred):
            total_loss = None
            for mask in self.mask_list:
            
                if not self.is_tanh:            
                    loss = (1.0 - tf.image.ssim (y_true*mask, y_pred*mask, 1.0)) / 2.0
                else:
                    loss = (1.0 - tf.image.ssim ( (y_true/2+0.5)*(mask/2+0.5), (y_pred/2+0.5)*(mask/2+0.5), 1.0)) / 2.0
                
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
                    
            return total_loss
            
    return DSSIMMaskLoss
    
def DSSIMPatchMaskLossClass(tf):
    class DSSIMPatchMaskLoss(object):
        def __init__(self, mask_list, is_tanh=False):
            self.mask_list = mask_list
            self.is_tanh = is_tanh
            
        def __call__(self,y_true, y_pred):
            total_loss = None
            for mask in self.mask_list:
                #import code
                #code.interact(local=dict(globals(), **locals()))
                
                y_true = tf.extract_image_patches ( y_true, (1,9,9,1), (1,1,1,1), (1,8,8,1), 'VALID' )        
                y_pred = tf.extract_image_patches ( y_pred, (1,9,9,1), (1,1,1,1), (1,8,8,1), 'VALID' ) 
                mask = tf.extract_image_patches ( tf.tile(mask,[1,1,1,3]) , (1,9,9,1), (1,1,1,1), (1,8,8,1), 'VALID' ) 
                if not self.is_tanh:            
                    loss = (1.0 - tf.image.ssim (y_true*mask, y_pred*mask, 1.0)) / 2.0
                else:
                    loss = (1.0 - tf.image.ssim ( (y_true/2+0.5)*(mask/2+0.5), (y_pred/2+0.5)*(mask/2+0.5), 1.0)) / 2.0
                
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
                    
            return total_loss
            
    return DSSIMPatchMaskLoss
    
def DSSIMLossClass(tf):
    class DSSIMLoss(object):
        def __init__(self, is_tanh=False):
            self.is_tanh = is_tanh
            
        def __call__(self,y_true, y_pred):
            if not self.is_tanh:            
                return (1.0 - tf.image.ssim (y_true, y_pred, 1.0)) / 2.0
            else:
                return (1.0 - tf.image.ssim ((y_true/2+0.5), (y_pred/2+0.5), 1.0)) / 2.0

    return DSSIMLoss
    
def rgb_to_lab(tf, rgb_input):
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
        #output [0, 100] , ~[-110, 110], ~[-110, 110]        
        lab_pixels = lab_pixels / tf.constant([100.0, 220.0, 220.0 ]) + tf.constant([0.0, 0.5, 0.5])
        #output [0-1, 0-1, 0-1]
        return tf.reshape(lab_pixels, tf.shape(rgb_input))
    
def lab_to_rgb(tf, lab):
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
        
def DSSIMPatchLossClass(tf, keras):
    class DSSIMPatchLoss(object):
        def __init__(self, is_tanh=False):
            self.is_tanh = is_tanh
            
        def __call__(self,y_true, y_pred):
            
            y_pred_lab = rgb_to_lab(tf, y_pred)
            y_true_lab = rgb_to_lab(tf, y_true)
           
            
            #import code
            #code.interact(local=dict(globals(), **locals()))
            
            return keras.backend.mean (  keras.backend.square(y_true_lab - y_pred_lab) ) # + (1.0 - tf.image.ssim (y_true, y_pred, 1.0)) / 2.0       
            
            if not self.is_tanh:            
                return (1.0 - tf.image.ssim (y_true, y_pred, 1.0)) / 2.0                
            else:
                return (1.0 - tf.image.ssim ((y_true/2+0.5), (y_pred/2+0.5), 1.0)) / 2.0
            
            #y_true_72 = tf.extract_image_patches ( y_true, (1,8,8,1), (1,1,1,1), (1,8,8,1), 'VALID' )        
            #y_pred_72 = tf.extract_image_patches ( y_pred, (1,8,8,1), (1,1,1,1), (1,8,8,1), 'VALID' ) 
                
            #y_true_36 = tf.extract_image_patches ( y_true, (1,8,8,1), (1,2,2,1), (1,8,8,1), 'VALID' )        
            #y_pred_36 = tf.extract_image_patches ( y_pred, (1,8,8,1), (1,2,2,1), (1,8,8,1), 'VALID' ) 
            
            #if not self.is_tanh:            
            #    return (1.0 - tf.image.ssim (y_true_72, y_pred_72, 1.0)) / 2.0 + \
            #           (1.0 - tf.image.ssim (y_true_36, y_pred_36, 1.0)) / 2.0
            #    
            #else:
            #    return (1.0 - tf.image.ssim ((y_true_72/2+0.5), (y_pred_72/2+0.5), 1.0)) / 2.0 + \
            #           (1.0 - tf.image.ssim ((y_true_36/2+0.5), (y_pred_36/2+0.5), 1.0)) / 2.0
                

    return DSSIMPatchLoss
    
def MSEMaskLossClass(keras):
    class MSEMaskLoss(object):
        def __init__(self, mask_list, is_tanh=False):
            self.mask_list = mask_list
            self.is_tanh = is_tanh
            
        def __call__(self,y_true, y_pred):
            K = keras.backend
            
            total_loss = None
            for mask in self.mask_list:
            
                if not self.is_tanh:            
                    loss = K.mean(K.square(y_true*mask - y_pred*mask))
                else:
                    loss = K.mean(K.square( (y_true/2+0.5)*(mask/2+0.5) - (y_pred/2+0.5)*(mask/2+0.5) ))
                
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
                    
            return total_loss
            
    return MSEMaskLoss
    
def PixelShufflerClass(keras):
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
                batch_size, c, h, w = input_shape
                if batch_size is None:
                    batch_size = -1
                rh, rw = self.size
                oh, ow = h * rh, w * rw
                oc = c // (rh * rw)

                out = keras.backend.reshape(inputs, (batch_size, rh, rw, oc, h, w))
                out = keras.backend.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
                out = keras.backend.reshape(out, (batch_size, oc, oh, ow))
                return out

            elif self.data_format == 'channels_last':
                batch_size, h, w, c = input_shape
                if batch_size is None:
                    batch_size = -1
                rh, rw = self.size
                oh, ow = h * rh, w * rw
                oc = c // (rh * rw)

                out = keras.backend.reshape(inputs, (batch_size, h, w, rh, rw, oc))
                out = keras.backend.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
                out = keras.backend.reshape(out, (batch_size, oh, ow, oc))
                return out

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
    return PixelShuffler

def conv(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    return x
    
def upscale(keras, input_tensor, filters, k_size=3):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters * 4, kernel_size=k_size, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    x = PixelShufflerClass(keras)()(x)
    return x
    
def upscale4(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters * 16, kernel_size=3, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    x = PixelShufflerClass(keras)(size=(4, 4))(x)
    return x
    
def res(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.convolutional.Conv2D(filters, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    return x
    
def resize_like(tf, keras, ref_tensor, input_tensor):
    def func(input_tensor, ref_tensor):
        H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
        return tf.image.resize_bilinear(input_tensor, [H.value, W.value])

    return keras.layers.Lambda(func, arguments={'ref_tensor':ref_tensor})(input_tensor)
    
def total_variation_loss(keras, x):
    K = keras.backend
    assert K.ndim(x) == 4    
    B,H,W,C = K.int_shape(x)
    a = K.square(x[:, :H - 1, :W - 1, :] - x[:, 1:, :W - 1, :])
    b = K.square(x[:, :H - 1, :W - 1, :] - x[:, :H - 1, 1:, :])
    
    return K.mean (a+b)
    

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
def UNet(keras, tf, input_shape, output_nc, num_downs, ngf=64, use_dropout=False):
    Conv2D = keras.layers.convolutional.Conv2D
    Conv2DTranspose = keras.layers.convolutional.Conv2DTranspose
    LeakyReLU = keras.layers.advanced_activations.LeakyReLU
    BatchNormalization = keras.layers.BatchNormalization
    ReLU = keras.layers.ReLU
    tanh = keras.layers.Activation('tanh')
    Dropout = keras.layers.Dropout
    Concatenate = keras.layers.Concatenate
    ZeroPadding2D = keras.layers.ZeroPadding2D
    
    conv_kernel_initializer = keras.initializers.RandomNormal(0, 0.02)
    norm_gamma_initializer = keras.initializers.RandomNormal(1, 0.02)
    
    input = keras.layers.Input (input_shape)
    
    def UNetSkipConnection(outer_nc, inner_nc, sub_model=None, outermost=False, innermost=False, use_dropout=False):       
        def func(inp):
            downconv_pad = ZeroPadding2D( (1,1) )
            downconv = Conv2D(inner_nc, kernel_size=4, kernel_initializer=conv_kernel_initializer, strides=2, padding='valid', use_bias=False)
            downrelu = LeakyReLU(0.2)        
            downnorm = BatchNormalization( gamma_initializer=norm_gamma_initializer )        
            
            upconv = Conv2DTranspose(outer_nc, kernel_size=4, kernel_initializer=conv_kernel_initializer, strides=2, padding='same', use_bias=False)
            uprelu = ReLU()
            upnorm = BatchNormalization( gamma_initializer=norm_gamma_initializer )
            
            if outermost:      
                x = inp
                x = downconv(downconv_pad(x))
                x = sub_model(x)
                x = uprelu(x)
                x = upconv(x)
                x = tanh(x)            
            elif innermost:           
                x = inp
                x = downrelu(x)
                x = downconv(downconv_pad(x))
                x = uprelu(x)
                x = upconv(x)
                x = upnorm(x)   
                x = Concatenate(axis=3)([inp, x])                                
            else:
                x = inp
                x = downrelu(x)
                x = downconv(downconv_pad(x))
                x = downnorm(x)
                x = sub_model(x)
                x = uprelu(x)                
                x = upconv(x)
                x = upnorm(x)                
                if use_dropout:
                    x = Dropout(0.5)(x)        
                x = Concatenate(axis=3)([inp, x])

            return x           
            
        return func
        

    unet_block = UNetSkipConnection(ngf * 8, ngf * 8, sub_model=None, innermost=True)

    #for i in range(num_downs - 5):
    #    unet_block = UNetSkipConnection(ngf * 8, ngf * 8, sub_model=unet_block, use_dropout=use_dropout)
    
    unet_block = UNetSkipConnection(ngf * 4  , ngf * 8, sub_model=unet_block)
    unet_block = UNetSkipConnection(ngf * 2  , ngf * 4, sub_model=unet_block)
    unet_block = UNetSkipConnection(ngf      , ngf * 2, sub_model=unet_block)
    unet_block = UNetSkipConnection(output_nc, ngf    , sub_model=unet_block, outermost=True)
    
    x = input
    x = unet_block(x)
    
    return keras.models.Model (input,x)

#predicts based on two past_image_tensors
def UNetTemporalPredictor(keras, tf, input_shape, output_nc, num_downs, ngf=32, use_dropout=False):
    K = keras.backend
    Conv2D = keras.layers.convolutional.Conv2D
    Conv2DTranspose = keras.layers.convolutional.Conv2DTranspose
    LeakyReLU = keras.layers.advanced_activations.LeakyReLU
    BatchNormalization = keras.layers.BatchNormalization
    ReLU = keras.layers.ReLU
    tanh = keras.layers.Activation('tanh')
    ReflectionPadding2D = ReflectionPadding2DClass(keras, tf)
    ZeroPadding2D = keras.layers.ZeroPadding2D
    Dropout = keras.layers.Dropout
    Concatenate = keras.layers.Concatenate
    
    conv_kernel_initializer = keras.initializers.RandomNormal(0, 0.02)
    norm_gamma_initializer = keras.initializers.RandomNormal(1, 0.02)

    past_2_image_tensor = keras.layers.Input (input_shape)
    past_1_image_tensor = keras.layers.Input (input_shape)
    
    def model1(input_shape):   
        input = keras.layers.Input (input_shape)
        x = input
        x = ReflectionPadding2D((3,3))(x)
        x = Conv2D(ngf, kernel_size=7, kernel_initializer=conv_kernel_initializer, strides=1, padding='valid', use_bias=False)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
        
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(ngf*2, kernel_size=3, kernel_initializer=conv_kernel_initializer, strides=1, padding='valid', use_bias=False)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
        
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(ngf*4, kernel_size=3, kernel_initializer=conv_kernel_initializer, strides=1, padding='valid', use_bias=False)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
        
        return keras.models.Model(input, x)
        
    def model3(input_shape):
        input = keras.layers.Input (input_shape)
        x = input
        
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(ngf*2, kernel_size=3, kernel_initializer=conv_kernel_initializer, strides=1, padding='valid', use_bias=False)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
        
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(ngf, kernel_size=3, kernel_initializer=conv_kernel_initializer, strides=1, padding='valid', use_bias=False)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
        
        x = ReflectionPadding2D((3,3))(x)
        x = Conv2D(output_nc, kernel_size=7, kernel_initializer=conv_kernel_initializer, strides=1, padding='valid', use_bias=False)(x)
        x = tanh(x)
        return keras.models.Model(input, x)
    
    x = Concatenate(axis=3)([ model1(input_shape)(past_2_image_tensor), model1(input_shape)(past_1_image_tensor) ])

    unet = UNet(keras, tf, K.int_shape(x)[1:], ngf*4, num_downs=num_downs, ngf=ngf*4*2, #ngf=ngf*4*4,
                        use_dropout=use_dropout)
                        
    x = unet(x)
    x = model3 ( K.int_shape(x)[1:] ) (x)

    return keras.models.Model ( [past_2_image_tensor,past_1_image_tensor], x )

def Resnet(keras, tf, input_shape, output_nc, ngf=64, use_dropout=False, n_blocks=6):
    Conv2D = keras.layers.convolutional.Conv2D
    Conv2DTranspose = keras.layers.convolutional.Conv2DTranspose
    LeakyReLU = keras.layers.advanced_activations.LeakyReLU
    BatchNormalization = keras.layers.BatchNormalization
    ReLU = keras.layers.ReLU
    Add = keras.layers.Add
    tanh = keras.layers.Activation('tanh')
    ReflectionPadding2D = ReflectionPadding2DClass(keras, tf)
    ZeroPadding2D = keras.layers.ZeroPadding2D
    Dropout = keras.layers.Dropout
    Concatenate = keras.layers.Concatenate
    
    conv_kernel_initializer = keras.initializers.RandomNormal(0, 0.02)
    norm_gamma_initializer = keras.initializers.RandomNormal(1, 0.02)
    use_bias = False

    input = keras.layers.Input (input_shape)
    
    def ResnetBlock(dim, use_dropout, use_bias):    
        def func(inp):
            x = inp
            
            x = ReflectionPadding2D((1,1))(x)
            x = Conv2D(dim, kernel_size=3, kernel_initializer=conv_kernel_initializer, padding='valid', use_bias=use_bias)(x)
            x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
            x = ReLU()(x)
            
            if use_dropout:
                x = Dropout(0.5)(x)
                
            x = ReflectionPadding2D((1,1))(x)
            x = Conv2D(dim, kernel_size=3, kernel_initializer=conv_kernel_initializer, padding='valid', use_bias=use_bias)(x)
            x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)    
            return Add()([x,inp])            
        return func

    x = input
    
    x = ReflectionPadding2D((3,3))(x)
    x = Conv2D(ngf, kernel_size=7, kernel_initializer=conv_kernel_initializer, padding='valid', use_bias=use_bias)(x)
    x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
    x = ReLU()(x)
    
    n_downsampling = 2
    for i in range(n_downsampling):            
        x = ZeroPadding2D( (1,1) ) (x)
        x = Conv2D(ngf * (2**i) * 2, kernel_size=3, kernel_initializer=conv_kernel_initializer, strides=2, padding='valid', use_bias=use_bias)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
    
    for i in range(n_blocks):
        x = ResnetBlock(ngf*(2**n_downsampling), use_dropout=use_dropout, use_bias=use_bias)(x)
    
    for i in range(n_downsampling):
        x = Conv2DTranspose( int(ngf* (2**(n_downsampling - i)) /2), kernel_size=3, kernel_initializer=conv_kernel_initializer, strides=2, padding='same', output_padding=1, use_bias=use_bias)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = ReLU()(x)
    
    
        
    x = ReflectionPadding2D((3,3))(x)
    x = Conv2D(output_nc, kernel_size=7, kernel_initializer=conv_kernel_initializer, padding='valid')(x)
    x = tanh(x)
    
    return keras.models.Model(input, x)
    
def NLayerDiscriminator(keras, tf, input_shape, ndf=64, n_layers=3, use_sigmoid=False):
    Conv2D = keras.layers.convolutional.Conv2D
    LeakyReLU = keras.layers.advanced_activations.LeakyReLU
    BatchNormalization = keras.layers.BatchNormalization
    sigmoid = keras.layers.Activation('sigmoid')
    ZeroPadding2D = keras.layers.ZeroPadding2D
    conv_kernel_initializer = keras.initializers.RandomNormal(0, 0.02)
    norm_gamma_initializer = keras.initializers.RandomNormal(1, 0.02)
    use_bias = False
    
    input = keras.layers.Input (input_shape, name="NLayerDiscriminatorInput") ###
    
    x = input
    x = ZeroPadding2D( (1,1) ) (x)
    x = Conv2D(ndf, kernel_size=4, kernel_initializer=conv_kernel_initializer, strides=2, padding='valid', use_bias=use_bias)(x)
    x = LeakyReLU(0.2)(x)
    
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
        nf_mult = min(2**n, 8)
        
        x = ZeroPadding2D( (1,1) ) (x)
        x = Conv2D(ndf * nf_mult, kernel_size=4, kernel_initializer=conv_kernel_initializer, strides=2, padding='valid', use_bias=use_bias)(x)
        x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
        x = LeakyReLU(0.2)(x)
        
    nf_mult = min(2**n_layers, 8)
    
    #x = ZeroPadding2D( (1,1) ) (x)
    x = Conv2D(ndf * nf_mult, kernel_size=4, kernel_initializer=conv_kernel_initializer, strides=1, padding='same', use_bias=use_bias)(x)
    x = BatchNormalization( gamma_initializer=norm_gamma_initializer )(x)
    x = LeakyReLU(0.2)(x)
    
    #x = ZeroPadding2D( (1,1) ) (x)
    x = Conv2D(1, kernel_size=4, kernel_initializer=conv_kernel_initializer, strides=1, padding='same', use_bias=use_bias)(x)
    
    if use_sigmoid:
        x = sigmoid(x)
        
    return keras.models.Model (input,x)

def ReflectionPadding2DClass(keras, tf):

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
            
    return ReflectionPadding2D