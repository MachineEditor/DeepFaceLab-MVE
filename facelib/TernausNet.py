import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from core.interact import interact as io
from core.leras import nn

class TernausNet(object):
    VERSION = 1

    def __init__ (self, name, resolution, load_weights=True, weights_file_root=None, training=False, place_model_on_cpu=False, run_on_cpu=False, optimizer=None, data_format="NHWC"):
        nn.initialize(data_format=data_format)
        tf = nn.tf

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent
        self.weights_file_root = weights_file_root

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.input_t  = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,3) )
            self.target_t = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,1) )

        # Initializing model classes
        with tf.device ('/CPU:0' if place_model_on_cpu else '/GPU:0'):
            self.net = nn.Ternaus(3, 64, name='Ternaus')
            self.net_weights = self.net.get_weights()

        model_name = f'{name}_{resolution}'             

        self.model_filename_list = [ [self.net, f'{model_name}.npy'] ]

        if training:
            if optimizer is None:
                raise ValueError("Optimizer should be provided for traning mode.")
                
            self.opt = optimizer
            self.opt.initialize_variables (self.net_weights, vars_on_cpu=place_model_on_cpu)
            self.model_filename_list += [ [self.opt, f'{model_name}_opt.npy' ] ]
        else:
            with tf.device ('/CPU:0' if run_on_cpu else '/GPU:0'):
                _, pred = self.net([self.input_t])
            
            def net_run(input_np):
                return nn.tf_sess.run ( [pred], feed_dict={self.input_t :input_np})[0]
            self.net_run = net_run

        # Loading/initializing all models/optimizers weights
        for model, filename in self.model_filename_list:
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
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
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
