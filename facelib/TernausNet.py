import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from core.interact import interact as io
from core.leras import nn

"""
Dataset used to train located in official DFL mega.nz folder
https://mega.nz/#F!b9MzCK4B!zEAG9txu7uaRUjXz9PtBqg

using https://github.com/ternaus/TernausNet
TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
"""

class TernausNet(object):
    VERSION = 1
    def __init__ (self, name, resolution, face_type_str, load_weights=True, weights_file_root=None, training=False, place_model_on_cpu=False):
        nn.initialize(data_format="NHWC")
        tf = nn.tf

        class Ternaus(nn.ModelBase):
            def on_build(self, in_ch, ch):

                self.features_0 = nn.Conv2D (in_ch, ch, kernel_size=3, padding='SAME')
                self.blurpool_0 = nn.BlurPool (filt_size=3)

                self.features_3 = nn.Conv2D (ch, ch*2, kernel_size=3, padding='SAME')
                self.blurpool_3 = nn.BlurPool (filt_size=3)

                self.features_6 = nn.Conv2D (ch*2, ch*4, kernel_size=3, padding='SAME')
                self.features_8 = nn.Conv2D (ch*4, ch*4, kernel_size=3, padding='SAME')
                self.blurpool_8 = nn.BlurPool (filt_size=3)

                self.features_11 = nn.Conv2D (ch*4, ch*8, kernel_size=3, padding='SAME')
                self.features_13 = nn.Conv2D (ch*8, ch*8, kernel_size=3, padding='SAME')
                self.blurpool_13 = nn.BlurPool (filt_size=3)

                self.features_16 = nn.Conv2D (ch*8, ch*8, kernel_size=3, padding='SAME')
                self.features_18 = nn.Conv2D (ch*8, ch*8, kernel_size=3, padding='SAME')
                self.blurpool_18 = nn.BlurPool (filt_size=3)

                self.conv_center = nn.Conv2D (ch*8, ch*8, kernel_size=3, padding='SAME')

                self.conv1_up = nn.Conv2DTranspose (ch*8, ch*4, kernel_size=3, padding='SAME')
                self.conv1 = nn.Conv2D (ch*12, ch*8, kernel_size=3, padding='SAME')

                self.conv2_up = nn.Conv2DTranspose (ch*8, ch*4, kernel_size=3, padding='SAME')
                self.conv2 = nn.Conv2D (ch*12, ch*8, kernel_size=3, padding='SAME')

                self.conv3_up = nn.Conv2DTranspose (ch*8, ch*2, kernel_size=3, padding='SAME')
                self.conv3 = nn.Conv2D (ch*6, ch*4, kernel_size=3, padding='SAME')

                self.conv4_up = nn.Conv2DTranspose (ch*4, ch, kernel_size=3, padding='SAME')
                self.conv4 = nn.Conv2D (ch*3, ch*2, kernel_size=3, padding='SAME')

                self.conv5_up = nn.Conv2DTranspose (ch*2, ch//2, kernel_size=3, padding='SAME')
                self.conv5 = nn.Conv2D (ch//2+ch, ch, kernel_size=3, padding='SAME')

                self.out_conv = nn.Conv2D (ch, 1, kernel_size=3, padding='SAME')

            def forward(self, inp):
                x, = inp

                x = x0 = tf.nn.relu(self.features_0(x))
                x = self.blurpool_0(x)

                x = x1 = tf.nn.relu(self.features_3(x))
                x = self.blurpool_3(x)

                x = tf.nn.relu(self.features_6(x))
                x = x2 = tf.nn.relu(self.features_8(x))
                x = self.blurpool_8(x)

                x = tf.nn.relu(self.features_11(x))
                x = x3 = tf.nn.relu(self.features_13(x))
                x = self.blurpool_13(x)

                x = tf.nn.relu(self.features_16(x))
                x = x4 = tf.nn.relu(self.features_18(x))
                x = self.blurpool_18(x)

                x = self.conv_center(x)

                x = tf.nn.relu(self.conv1_up(x))
                x = tf.concat( [x,x4], -1)
                x = tf.nn.relu(self.conv1(x))

                x = tf.nn.relu(self.conv2_up(x))
                x = tf.concat( [x,x3], -1)
                x = tf.nn.relu(self.conv2(x))

                x = tf.nn.relu(self.conv3_up(x))
                x = tf.concat( [x,x2], -1)
                x = tf.nn.relu(self.conv3(x))

                x = tf.nn.relu(self.conv4_up(x))
                x = tf.concat( [x,x1], -1)
                x = tf.nn.relu(self.conv4(x))

                x = tf.nn.relu(self.conv5_up(x))
                x = tf.concat( [x,x0], -1)
                x = tf.nn.relu(self.conv5(x))

                x = tf.nn.sigmoid(self.out_conv(x))
                return x

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent
        self.weights_path = weights_file_root / ('%s_%d_%s.npy' % (name, resolution, face_type_str) )

        e = tf.device('/CPU:0') if place_model_on_cpu else None

        if e is not None: e.__enter__()
        self.net = Ternaus(3, 64, name='Ternaus')
        if load_weights:
            self.net.load_weights (self.weights_path)
        else:
            self.net.init_weights()
        if e is not None: e.__exit__(None,None,None)

        self.net.build_for_run ( [(tf.float32, nn.get4Dshape (resolution,resolution,3) )] )

        if training:
            raise Exception("training not supported yet")


            """
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
            """


        """
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
        """

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.net.save_weights (str(self.weights_path))

    def train(self, inp, real):
        loss, = self.train_func ([inp, real])
        return loss

    def extract (self, input_image):
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        result = np.clip ( self.net.run([input_image]), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result
