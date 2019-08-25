import numpy as np
import cv2
from pathlib import Path
from nnlib import nnlib
from interact import interact as io

class RankSRGAN():
    def __init__(self):
        exec( nnlib.import_all(), locals(), globals() )

        class PixelShufflerTorch(KL.Layer):
            def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
                super(PixelShufflerTorch, self).__init__(**kwargs)
                self.data_format = data_format
                self.size = size

            def call(self, inputs):
                input_shape = K.shape(inputs)
                if K.int_shape(input_shape)[0] != 4:
                    raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

                batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], K.int_shape(inputs)[-1]
                rh, rw = self.size
                oh, ow = h * rh, w * rw
                oc = c // (rh * rw)

                out = inputs
                out = K.permute_dimensions(out, (0, 3, 1, 2)) #NCHW

                out = K.reshape(out, (batch_size, oc, rh, rw, h, w))
                out = K.permute_dimensions(out, (0, 1, 4, 2, 5, 3))
                out = K.reshape(out, (batch_size, oc, oh, ow))

                out = K.permute_dimensions(out, (0, 2, 3, 1))
                return out

            def compute_output_shape(self, input_shape):
                if len(input_shape) != 4:
                    raise ValueError('Inputs should have rank ' + str(4) + '; Received input shape:', str(input_shape))

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
                base_config = super(PixelShufflerTorch, self).get_config()

                return dict(list(base_config.items()) + list(config.items()))

        def res_block(inp, name_prefix):
            x = inp
            x = Conv2D (ndf, kernel_size=3, strides=1, padding='same', activation="relu", name=name_prefix+"0")(x)
            x = Conv2D (ndf, kernel_size=3, strides=1, padding='same', name=name_prefix+"2")(x)
            return Add()([inp,x])

        ndf = 64
        nb = 16
        inp = Input ( (None, None,3) )
        x = inp

        x = x0 = Conv2D (ndf, kernel_size=3, strides=1, padding='same', name="model0")(x)
        for i in range(nb):
            x = res_block(x, "model1%.2d" %i )
        x = Conv2D (ndf, kernel_size=3, strides=1, padding='same', name="model1160")(x)
        x = Add()([x0,x])

        x = ReLU() ( PixelShufflerTorch() ( Conv2D (ndf*4, kernel_size=3, strides=1, padding='same', name="model2")(x) ) )
        x = ReLU() ( PixelShufflerTorch() ( Conv2D (ndf*4, kernel_size=3, strides=1, padding='same', name="model5")(x) ) )

        x = Conv2D (ndf, kernel_size=3, strides=1, padding='same', activation="relu", name="model8")(x)
        x = Conv2D (3,   kernel_size=3, strides=1, padding='same', name="model10")(x)
        self.model = Model(inp, x )
        self.model.load_weights ( Path(__file__).parent / 'RankSRGAN.h5')

    def upscale(self, img, scale=2, is_bgr=True, is_float=True):
        if scale not in [2,4]:
            raise ValueError ("RankSRGAN: supported scale are 2 or 4.")

        if not is_bgr:
            img = img[...,::-1]

        if not is_float:
            img /= 255.0

        h, w = img.shape[:2]
        ch = img.shape[2] if len(img.shape) >= 3 else 1

        output = self.model.predict([img[None,...]])[0]

        if scale == 2:
            output = cv2.resize (output, (w*scale, h*scale), cv2.INTER_CUBIC)

        if not is_float:
            output = np.clip (output * 255.0, 0, 255.0)

        if not is_bgr:
            output = output[...,::-1]

        return output