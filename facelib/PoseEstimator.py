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
"""

class PoseEstimator(object):
    VERSION = 1
    def __init__ (self, resolution, face_type_str, load_weights=True, weights_file_root=None, training=False):
        exec( nnlib.import_all(), locals(), globals() )

        self.class_num = 91
        
        self.model = PoseEstimator.BuildModel(resolution, class_num=self.class_num)

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent

        self.weights_path = weights_file_root / ('PoseEst_%d_%s.h5' % (resolution, face_type_str) )

        if load_weights:
            self.model.load_weights (str(self.weights_path))

        idx_tensor = np.array([idx for idx in range(self.class_num)], dtype=K.floatx() )
        idx_tensor = K.constant(idx_tensor)

        inp_t, = self.model.inputs
        pitch_bins_t, yaw_bins_t, roll_bins_t = self.model.outputs
        
        pitch_t, yaw_t, roll_t = K.sum ( pitch_bins_t * idx_tensor, 1), K.sum ( yaw_bins_t * idx_tensor, 1), K.sum ( roll_bins_t * idx_tensor, 1)
        
        inp_pitch_bins_t = Input ( (self.class_num,) )
        inp_pitch_t = Input ( (1,) )
        
        inp_yaw_bins_t = Input ( (self.class_num,) )
        inp_yaw_t = Input ( (1,) )
        
        inp_roll_bins_t = Input ( (self.class_num,) )
        inp_roll_t = Input ( (1,) )
        
        alpha = 0.001

        pitch_loss = K.categorical_crossentropy(inp_pitch_bins_t, pitch_bins_t) \
                        + alpha * K.mean(K.square( inp_pitch_t - pitch_t), -1)
        
        yaw_loss = K.categorical_crossentropy(inp_yaw_bins_t, yaw_bins_t) \
                        + alpha * K.mean(K.square( inp_yaw_t - yaw_t), -1)
                        
        roll_loss = K.categorical_crossentropy(inp_roll_bins_t, roll_bins_t) \
                        + alpha * K.mean(K.square( inp_roll_t - roll_t), -1)
        
        
        loss = K.mean( pitch_loss + yaw_loss + roll_loss )
    
        opt = Adam(lr=0.001, tf_cpu_mode=2)
        
        if training:
            self.train = K.function ([inp_t, inp_pitch_bins_t, inp_pitch_t, inp_yaw_bins_t, inp_yaw_t, inp_roll_bins_t, inp_roll_t],
                                     [loss], opt.get_updates(loss, self.model.trainable_weights) )

        self.view = K.function ([inp_t], [pitch_t, yaw_t, roll_t] )
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.model.save_weights (str(self.weights_path))

    def train_on_batch(self, imgs, pitch_yaw_roll):
        c = ( (pitch_yaw_roll+1) * 45.0 ).astype(np.int).astype(K.floatx())
        
        inp_pitch = c[:,0:1]
        inp_yaw = c[:,1:2]
        inp_roll = c[:,2:3]

        inp_pitch_bins = keras.utils.to_categorical(inp_pitch, self.class_num )
        inp_yaw_bins = keras.utils.to_categorical(inp_yaw, self.class_num )
        inp_roll_bins = keras.utils.to_categorical(inp_roll, self.class_num )

        loss, = self.train( [imgs, inp_pitch_bins, inp_pitch, inp_yaw_bins, inp_yaw, inp_roll_bins, inp_roll] )

        return loss

    def extract (self, input_image, is_input_tanh=False):
        if is_input_tanh:
            raise NotImplemented("is_input_tanh")
            
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        pitch, yaw, roll = self.view( [input_image] )
        result = np.concatenate( (pitch[...,np.newaxis], yaw[...,np.newaxis], roll[...,np.newaxis]), -1 )
        result = np.clip ( result / 45.0 - 1, -1.0, 1.0 )

        if input_shape_len == 3:
            result = result[0]

        return result

    @staticmethod
    def BuildModel ( resolution, class_num):
        exec( nnlib.import_all(), locals(), globals() )
        inp = Input ( (resolution,resolution,3) )
        x = inp
        x = PoseEstimator.Flow(class_num=class_num)(x)
        model = Model(inp,x)
        return model

    @staticmethod
    def Flow(class_num):
        exec( nnlib.import_all(), locals(), globals() )

        def func(input):
            x = input
            
            # resnet50 = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=K.int_shape(x)[1:], pooling='avg')
            # x = resnet50(x)
            # pitch = Dense(class_num, activation='softmax', name='pitch')(x)
            # yaw = Dense(class_num, activation='softmax', name='yaw')(x)
            # roll = Dense(class_num, activation='softmax', name='roll')(x)
            
            # return [pitch, yaw, roll]
            
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
            
            pitch = Dense(class_num, activation='softmax', name='pitch')(x)
            yaw = Dense(class_num, activation='softmax', name='yaw')(x)
            roll = Dense(class_num, activation='softmax', name='roll')(x)
            
            return [pitch, yaw, roll]

        return func
