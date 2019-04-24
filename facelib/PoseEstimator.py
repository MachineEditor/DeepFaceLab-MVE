import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from interact import interact as io
from nnlib import nnlib

"""
PoseEstimator estimates pitch, yaw, roll, from FAN aligned face.
trained on https://www.umdfaces.io
based on https://arxiv.org/pdf/1901.06778.pdf HYBRID COARSE-FINE CLASSIFICATION FOR HEAD POSE ESTIMATION  
"""

class PoseEstimator(object):
    VERSION = 1
    def __init__ (self, resolution, face_type_str, load_weights=True, weights_file_root=None, training=False):
        exec( nnlib.import_all(), locals(), globals() )

        self.angles = [90, 45, 30, 10, 2]
        self.alpha_cat_losses = [7,5,3,1,1]
        self.class_nums = [ angle+1 for angle in self.angles ]
        
        self.model = PoseEstimator.BuildModel(resolution, class_nums=self.class_nums)
        

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent

        self.weights_path = weights_file_root / ('PoseEst_%d_%s.h5' % (resolution, face_type_str) )

        if load_weights:
            self.model.load_weights (str(self.weights_path))
        else:
            conv_weights_list = []
            for layer in self.model.layers:
                if type(layer) == keras.layers.Conv2D:
                    conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights            
            CAInitializerMP ( conv_weights_list )
            
        inp_t, = self.model.inputs
        bins_t = self.model.outputs
        
        inp_pitch_t = Input ( (1,) )
        inp_yaw_t = Input ( (1,) )
        inp_roll_t = Input ( (1,) )
        
        inp_bins_t = []
        for class_num in self.class_nums:
            inp_bins_t += [ Input ((class_num,)), Input ((class_num,)), Input ((class_num,)) ]
        
        loss_pitch = []
        loss_yaw = []
        loss_roll = []
        
        for i,class_num in enumerate(self.class_nums):
            a = self.alpha_cat_losses[i]
            loss_pitch += [ a*K.categorical_crossentropy( inp_bins_t[i*3+0], bins_t[i*3+0] ) ]
            loss_yaw   += [ a*K.categorical_crossentropy( inp_bins_t[i*3+1], bins_t[i*3+1] ) ]
            loss_roll  += [ a*K.categorical_crossentropy( inp_bins_t[i*3+2], bins_t[i*3+2] ) ]
            
        idx_tensor =  K.constant( np.array([idx for idx in range(self.class_nums[0])], dtype=K.floatx() ) )
        pitch_t, yaw_t, roll_t = K.sum ( bins_t[0] * idx_tensor, 1), K.sum ( bins_t[1] * idx_tensor, 1), K.sum ( bins_t[2] * idx_tensor, 1)
        
        reg_alpha = 2
        reg_pitch_loss = reg_alpha * K.mean(K.square( inp_pitch_t - pitch_t), -1)        
        reg_yaw_loss = reg_alpha   * K.mean(K.square( inp_yaw_t - yaw_t), -1)                        
        reg_roll_loss =  reg_alpha * K.mean(K.square( inp_roll_t - roll_t), -1)
                
        pitch_loss = reg_pitch_loss + sum(loss_pitch)
        yaw_loss   = reg_yaw_loss   + sum(loss_yaw)
        roll_loss  = reg_roll_loss  + sum(loss_roll)
    
        opt = Adam(lr=0.000001)
        
        if training:
            self.train = K.function ([inp_t, inp_pitch_t, inp_yaw_t, inp_roll_t] + inp_bins_t,
                                     [K.mean(pitch_loss),K.mean(yaw_loss),K.mean(roll_loss)], opt.get_updates( [pitch_loss,yaw_loss,roll_loss], self.model.trainable_weights) )

        self.view = K.function ([inp_t], [pitch_t, yaw_t, roll_t] )
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.model.save_weights (str(self.weights_path))

    def train_on_batch(self, imgs, pitch_yaw_roll):
        pyr = pitch_yaw_roll+1

        feed = [imgs]
        
        for i, (angle, class_num) in enumerate(zip(self.angles, self.class_nums)):
            c = np.round(pyr * (angle / 2) ).astype(K.floatx())
            inp_pitch = c[:,0:1]
            inp_yaw = c[:,1:2]
            inp_roll = c[:,2:3]
            if i == 0:
                feed += [inp_pitch, inp_yaw, inp_roll]
            
            inp_pitch_bins = keras.utils.to_categorical(inp_pitch, class_num )
            inp_yaw_bins = keras.utils.to_categorical(inp_yaw, class_num )
            inp_roll_bins = keras.utils.to_categorical(inp_roll, class_num )
            feed += [inp_pitch_bins, inp_yaw_bins, inp_roll_bins] 
            #import code
            #code.interact(local=dict(globals(), **locals()))

        pitch_loss,yaw_loss,roll_loss = self.train(feed)
        return pitch_loss,yaw_loss,roll_loss

    def extract (self, input_image, is_input_tanh=False):
        if is_input_tanh:
            raise NotImplemented("is_input_tanh")
            
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        pitch, yaw, roll = self.view( [input_image] )
        result = np.concatenate( (pitch[...,np.newaxis], yaw[...,np.newaxis], roll[...,np.newaxis]), -1 )
        result = np.clip ( result / (self.angles[0] / 2) - 1, -1.0, 1.0 )

        if input_shape_len == 3:
            result = result[0]

        return result

    @staticmethod
    def BuildModel ( resolution, class_nums):
        exec( nnlib.import_all(), locals(), globals() )
        inp = Input ( (resolution,resolution,3) )
        x = inp
        x = PoseEstimator.Flow(class_nums=class_nums)(x)
        model = Model(inp,x)
        return model

    @staticmethod
    def Flow(class_nums):
        exec( nnlib.import_all(), locals(), globals() )

        def func(input):
            x = input
            
            # resnet50 = keras.applications.ResNet50(include_top=False, weights=None, input_shape=K.int_shape(x)[1:], pooling='avg')
            # x = resnet50(x)
            # output = []
            # for class_num in class_nums:
            #     pitch = Dense(class_num, activation='softmax')(x)
            #     yaw = Dense(class_num, activation='softmax')(x)
            #     roll = Dense(class_num, activation='softmax')(x)
            #     output += [pitch,yaw,roll]
                
            # return output
            
            x = Conv2D(64, kernel_size=11, strides=4, padding='same', activation='relu')(x)
            x = MaxPooling2D( (3,3), strides=2 )(x)

            x = Conv2D(192, kernel_size=5, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D( (3,3), strides=2 )(x)

            x = Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = MaxPooling2D( (3,3), strides=2 )(x)
            
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            
            output = []
            for class_num in class_nums:
                pitch = Dense(class_num, activation='softmax')(x)
                yaw = Dense(class_num, activation='softmax')(x)
                roll = Dense(class_num, activation='softmax')(x)
                output += [pitch,yaw,roll]
                
            return output

        return func
