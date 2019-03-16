import numpy as np
import os
import cv2
from pathlib import Path
from nnlib import nnlib

class FANSegmentator(object):
    def __init__ (self, resolution, face_type_str, load_weights=True, weights_file_root=None):
        exec( nnlib.import_all(), locals(), globals() )
        
        self.model = FANSegmentator.BuildModel(resolution, ngf=32)
        
        if weights_file_root:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent
            
        self.weights_path = weights_file_root / ('FANSeg_%d_%s.h5' % (resolution, face_type_str) )
        
        if load_weights:
            self.model.load_weights (str(self.weights_path))
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level
        
    def save_weights(self):
        self.model.save_weights (str(self.weights_path))
        
    def train_on_batch(self, inp, outp):
        return self.model.train_on_batch(inp, outp)
        
    def extract_from_bgr (self, input_image):
        #return np.clip ( self.model.predict(input_image), 0, 1.0 )
        return np.clip ( (self.model.predict(input_image) + 1) / 2.0, 0, 1.0 )
        
    @staticmethod
    def BuildModel ( resolution, ngf=64):
        exec( nnlib.import_all(), locals(), globals() )
        inp = Input ( (resolution,resolution,3) )
        x = inp
        x = FANSegmentator.EncFlow(ngf=ngf)(x)
        x = FANSegmentator.DecFlow(ngf=ngf)(x)
        model = Model(inp,x)
        model.compile (loss='mse', optimizer=Padam(tf_cpu_mode=2) )
        #model.compile (loss='mse', optimizer=Adam(tf_cpu_mode=2) )
        return model
        
    @staticmethod
    def EncFlow(ngf=64, num_downs=4):
        exec( nnlib.import_all(), locals(), globals() )

        use_bias = True
        def XNormalization(x):
            return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(XNormalization(Conv2D(dim, kernel_size=5, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)))
            return func 
            
        def func(input):     
            x = input
            
            result = []
            for i in range(num_downs):
                x = downscale ( min(ngf*(2**i), ngf*8) )(x)
                result += [x]                
                
            return result
        return func
        
    @staticmethod
    def DecFlow(output_nc=1, ngf=64, activation='tanh'):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = True
        def XNormalization(x):
            return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
            
        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()( LeakyReLU(0.1)(XNormalization(Conv2D(dim, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x))))
            return func 
            
        def func(input):
            input_len = len(input)
            
            x = input[input_len-1]
            for i in range(input_len-1, -1, -1):          
                x = upscale( min(ngf* (2**i) *4, ngf*8 *4 ) )(x)
                if i != 0:
                    x = Concatenate(axis=3)([ input[i-1] , x])
        
            return Conv2D(output_nc, 3, 1, 'same', activation=activation)(x)
        return func