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