import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from interact import interact as io
from nnlib import nnlib

"""
Dataset used to train located in official DFL mega.nz folder
https://mega.nz/#F!b9MzCK4B!zEAG9txu7uaRUjXz9PtBqg

using https://github.com/ternaus/TernausNet
TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
"""

class TernausNet(object):
    VERSION = 1
    def __init__ (self, name, resolution, face_type_str, load_weights=True, weights_file_root=None, training=False):
        exec( nnlib.import_all(), locals(), globals() )

        self.model = TernausNet.BuildModel(resolution, ngf=64)

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent

        self.weights_path = weights_file_root / ('%s_%d_%s.h5' % (name, resolution, face_type_str) )

        if load_weights:
            self.model.load_weights (str(self.weights_path))
        else:
            if training:
                try:
                    with open( Path(__file__).parent / 'vgg11_enc_weights.npy', 'rb' ) as f:
                        d = pickle.loads (f.read())

                    for i in [0,3,6,8,11,13,16,18]:
                        s = 'features.%d' % i

                        self.model.get_layer (s).set_weights ( d[s] )
                except:
                    io.log_err("Unable to load VGG11 pretrained weights from vgg11_enc_weights.npy")

                conv_weights_list = []
                for layer in self.model.layers:
                    if 'CA.' in layer.name:
                        conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights
                CAInitializerMP ( conv_weights_list )

        if training:
            inp_t = Input ( (resolution, resolution, 3) )
            real_t = Input ( (resolution, resolution, 1) )
            out_t = self.model(inp_t)

            loss = K.mean(10*K.binary_crossentropy(real_t,out_t) )

            out_t_diff1 = out_t[:, 1:, :, :] - out_t[:, :-1, :, :]
            out_t_diff2 = out_t[:, :, 1:, :] - out_t[:, :, :-1, :]

            total_var_loss = K.mean( 0.1*K.abs(out_t_diff1), axis=[1, 2, 3] ) + K.mean( 0.1*K.abs(out_t_diff2), axis=[1, 2, 3] )

            opt = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)

            self.train_func = K.function  ( [inp_t, real_t], [K.mean(loss)], opt.get_updates( [loss], self.model.trainable_weights) )


    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.model.save_weights (str(self.weights_path))

    def train(self, inp, real):
        loss, = self.train_func ([inp, real])
        return loss

    def extract (self, input_image, is_input_tanh=False):
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        result = np.clip ( self.model.predict( [input_image] ), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result

    @staticmethod
    def BuildModel ( resolution, ngf=64):
        exec( nnlib.import_all(), locals(), globals() )
        inp = Input ( (resolution,resolution,3) )
        x = inp
        x = TernausNet.Flow(ngf=ngf)(x)
        model = Model(inp,x)
        return model

    @staticmethod
    def Flow(ngf=64):
        exec( nnlib.import_all(), locals(), globals() )

        def func(input):
            x = input

            x0 = x = Conv2D(ngf, kernel_size=3, strides=1, padding='same', activation='relu', name='features.0')(x)
            x = BlurPool(filt_size=3)(x)

            x1 = x = Conv2D(ngf*2, kernel_size=3, strides=1, padding='same', activation='relu', name='features.3')(x)
            x = BlurPool(filt_size=3)(x)

            x = Conv2D(ngf*4, kernel_size=3, strides=1, padding='same', activation='relu', name='features.6')(x)
            x2 = x = Conv2D(ngf*4, kernel_size=3, strides=1, padding='same', activation='relu', name='features.8')(x)
            x = BlurPool(filt_size=3)(x)

            x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.11')(x)
            x3 = x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.13')(x)
            x = BlurPool(filt_size=3)(x)

            x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.16')(x)
            x4 = x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.18')(x)
            x = BlurPool(filt_size=3)(x)

            x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', name='CA.1')(x)

            x = Conv2DTranspose (ngf*4, 3, strides=2, padding='same', activation='relu', name='CA.2') (x)
            x = Concatenate(axis=3)([ x, x4])
            x = Conv2D (ngf*8, 3, strides=1, padding='same', activation='relu', name='CA.3') (x)

            x = Conv2DTranspose (ngf*4, 3, strides=2, padding='same', activation='relu', name='CA.4') (x)
            x = Concatenate(axis=3)([ x, x3])
            x = Conv2D (ngf*8, 3, strides=1, padding='same', activation='relu', name='CA.5') (x)

            x = Conv2DTranspose (ngf*2, 3, strides=2, padding='same', activation='relu', name='CA.6') (x)
            x = Concatenate(axis=3)([ x, x2])
            x = Conv2D (ngf*4, 3, strides=1, padding='same', activation='relu', name='CA.7') (x)

            x = Conv2DTranspose (ngf, 3, strides=2, padding='same', activation='relu', name='CA.8') (x)
            x = Concatenate(axis=3)([ x, x1])
            x = Conv2D (ngf*2, 3, strides=1, padding='same', activation='relu', name='CA.9') (x)

            x = Conv2DTranspose (ngf // 2, 3, strides=2, padding='same', activation='relu', name='CA.10') (x)
            x = Concatenate(axis=3)([ x, x0])
            x = Conv2D (ngf, 3, strides=1, padding='same', activation='relu', name='CA.11') (x)

            return Conv2D(1, 3, strides=1, padding='same', activation='sigmoid', name='CA.12')(x)


        return func
