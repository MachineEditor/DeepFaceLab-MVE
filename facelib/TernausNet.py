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
            def on_build(self, in_ch, base_ch):

                self.features_0 = nn.Conv2D (in_ch, base_ch, kernel_size=3, padding='SAME')
                self.blurpool_0 = nn.BlurPool (filt_size=3)

                self.features_3 = nn.Conv2D (base_ch, base_ch*2, kernel_size=3, padding='SAME')
                self.blurpool_3 = nn.BlurPool (filt_size=3)

                self.features_6 = nn.Conv2D (base_ch*2, base_ch*4, kernel_size=3, padding='SAME')
                self.features_8 = nn.Conv2D (base_ch*4, base_ch*4, kernel_size=3, padding='SAME')
                self.blurpool_8 = nn.BlurPool (filt_size=3)

                self.features_11 = nn.Conv2D (base_ch*4, base_ch*8, kernel_size=3, padding='SAME')
                self.features_13 = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')
                self.blurpool_13 = nn.BlurPool (filt_size=3)

                self.features_16 = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')
                self.features_18 = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')
                self.blurpool_18 = nn.BlurPool (filt_size=3)

                self.conv_center = nn.Conv2D (base_ch*8, base_ch*8, kernel_size=3, padding='SAME')

                self.conv1_up = nn.Conv2DTranspose (base_ch*8, base_ch*4, kernel_size=3, padding='SAME')
                self.conv1 = nn.Conv2D (base_ch*12, base_ch*8, kernel_size=3, padding='SAME')

                self.conv2_up = nn.Conv2DTranspose (base_ch*8, base_ch*4, kernel_size=3, padding='SAME')
                self.conv2 = nn.Conv2D (base_ch*12, base_ch*8, kernel_size=3, padding='SAME')

                self.conv3_up = nn.Conv2DTranspose (base_ch*8, base_ch*2, kernel_size=3, padding='SAME')
                self.conv3 = nn.Conv2D (base_ch*6, base_ch*4, kernel_size=3, padding='SAME')

                self.conv4_up = nn.Conv2DTranspose (base_ch*4, base_ch, kernel_size=3, padding='SAME')
                self.conv4 = nn.Conv2D (base_ch*3, base_ch*2, kernel_size=3, padding='SAME')

                self.conv5_up = nn.Conv2DTranspose (base_ch*2, base_ch//2, kernel_size=3, padding='SAME')
                self.conv5 = nn.Conv2D (base_ch//2+base_ch, base_ch, kernel_size=3, padding='SAME')

                self.out_conv = nn.Conv2D (base_ch, 1, kernel_size=3, padding='SAME')

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

                logits = self.out_conv(x)
                return logits, tf.nn.sigmoid(logits)

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent
        self.weights_file_root = weights_file_root

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.input_t  = tf.placeholder (nn.tf_floatx, nn.get4Dshape(resolution,resolution,3) )
            self.target_t = tf.placeholder (nn.tf_floatx, nn.get4Dshape(resolution,resolution,1) )

        # Initializing model classes
        with tf.device ('/CPU:0' if place_model_on_cpu else '/GPU:0'):
            self.net = Ternaus(3, 64, name='Ternaus')
            self.net_weights = self.net.get_weights()

        self.model_filename_list = [ [self.net, '%s_%d_%s.npy' % (name, resolution, face_type_str) ] ]

        if training:
            self.opt = nn.TFRMSpropOptimizer(lr=0.0001, name='opt')
            self.opt.initialize_variables (self.net_weights, vars_on_cpu=place_model_on_cpu)
            self.model_filename_list += [ [self.opt, '%s_%d_%s_opt.npy' % (name, resolution, face_type_str) ] ]
        else:
            _, pred = self.net([self.input_t])
            def net_run(input_np):
                return nn.tf_sess.run ( [pred], feed_dict={self.input_t :input_np})[0]
            self.net_run = net_run

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = not load_weights

            if not do_init:
                do_init = not model.load_weights( self.weights_file_root / filename )

            if do_init:
                model.init_weights()
                if model == self.net:
                    try:
                        with open( Path(__file__).parent / 'vgg11_enc_weights.npy', 'rb' ) as f:
                            d = pickle.loads (f.read())

                        for i in [0,3,6,8,11,13,16,18]:
                            model.get_layer_by_name ('features_%d' % i).set_weights ( d['features.%d' % i] )
                    except:
                        io.log_err("Unable to load VGG11 pretrained weights from vgg11_enc_weights.npy")

    def save_weights(self):
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving"):
            model.save_weights( self.weights_file_root / filename )

    def extract (self, input_image):
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None,...]

        result = np.clip ( self.net_run(input_image), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result

"""
if load_weights:
    self.net.load_weights (self.weights_path)
else:
    self.net.init_weights()

if load_weights:
    self.opt.load_weights (self.opt_path)
else:
    self.opt.init_weights()
"""
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
