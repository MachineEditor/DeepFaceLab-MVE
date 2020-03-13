import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from core.interact import interact as io
from core.leras import nn


class DFLSegNet(object):
    VERSION = 1

    def __init__ (self, name, 
                        resolution, 
                        load_weights=True, 
                        weights_file_root=None, 
                        training=False, 
                        place_model_on_cpu=False, 
                        run_on_cpu=False, 
                        optimizer=None, 
                        data_format="NHWC"):
                        
        nn.initialize(data_format=data_format)
        tf = nn.tf

        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.input_t  = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,3) )
            self.target_t = tf.placeholder (nn.floatx, nn.get4Dshape(resolution,resolution,1) )

        # Initializing model classes
        archi = nn.DFLSegnetArchi()
        with tf.device ('/CPU:0' if place_model_on_cpu else '/GPU:0'):
            self.enc = archi.Encoder(3, 64, name='Encoder')
            self.dec = archi.Decoder(64, 1, name='Decoder')
            self.enc_dec_weights = self.enc.get_weights()+self.dec.get_weights()

        model_name = f'{name}_{resolution}'

        self.model_filename_list = [ [self.enc, f'{model_name}_enc.npy'],
                                     [self.dec, f'{model_name}_dec.npy'],
                                    ]

        if training:
            if optimizer is None:
                raise ValueError("Optimizer should be provided for training mode.")

            self.opt = optimizer
            self.opt.initialize_variables (self.enc_dec_weights, vars_on_cpu=place_model_on_cpu)
            self.model_filename_list += [ [self.opt, f'{model_name}_opt.npy' ] ]
        else:
            with tf.device ('/CPU:0' if run_on_cpu else '/GPU:0'):
                _, pred = self.dec(self.enc(self.input_t))

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

    def flow(self, x):
        return self.dec(self.enc(x))

    def get_weights(self):
        return self.enc_dec_weights

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