from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2

from nnlib import DSSIMMaskLossClass
from nnlib import conv
from nnlib import upscale
from facelib import FaceType

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoderH5 = 'decoder.h5'
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'

    #override
    def onInitialize(self, **in_options):
        self.set_vram_batch_requirements( {4.5:4,5:4,6:8,7:12,8:16,9:20,10:24,11:24,12:32,13:48} )

        ae_input_layer = self.keras.layers.Input(shape=(128, 128, 3))
        mask_layer = self.keras.layers.Input(shape=(128, 128, 1)) #same as output

        self.encoder = self.Encoder(ae_input_layer)
        self.decoder = self.Decoder()
        self.inter_B = self.Intermediate ()
        self.inter_AB = self.Intermediate ()
        
        if not self.is_first_run():
            self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder.load_weights  (self.get_strpath_storage_for_file(self.decoderH5))
            self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
            self.inter_AB.load_weights (self.get_strpath_storage_for_file(self.inter_ABH5))

        code = self.encoder(ae_input_layer)
        AB = self.inter_AB(code)
        B = self.inter_B(code)
        self.autoencoder_src = self.keras.models.Model([ae_input_layer,mask_layer], self.decoder(self.keras.layers.Concatenate()([AB, AB])) )
        self.autoencoder_dst = self.keras.models.Model([ae_input_layer,mask_layer], self.decoder(self.keras.layers.Concatenate()([B, AB])) )
            
        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
                
        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        dssimloss = DSSIMMaskLossClass(self.tf)([mask_layer])
        self.autoencoder_src.compile(optimizer=optimizer, loss=[dssimloss, 'mse'] )
        self.autoencoder_dst.compile(optimizer=optimizer, loss=[dssimloss, 'mse'] )
  
        if self.is_training_mode:
            from models import TrainingDataGenerator
            f = TrainingDataGenerator.SampleTypeFlags 
            self.set_training_data_generators ([            
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ [f.WARPED_TRANSFORMED | f.FULL_FACE | f.MODE_BGR, 128], [f.TRANSFORMED | f.FULL_FACE | f.MODE_BGR, 128], [f.TRANSFORMED | f.FULL_FACE | f.MODE_M | f.MASK_FULL, 128] ], random_flip=True ),
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ [f.WARPED_TRANSFORMED | f.FULL_FACE | f.MODE_BGR, 128], [f.TRANSFORMED | f.FULL_FACE | f.MODE_BGR, 128], [f.TRANSFORMED | f.FULL_FACE | f.MODE_M | f.MASK_FULL, 128] ], random_flip=True )
                ])
            
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder, self.get_strpath_storage_for_file(self.decoderH5)],
                                [self.inter_B, self.get_strpath_storage_for_file(self.inter_BH5)],
                                [self.inter_AB, self.get_strpath_storage_for_file(self.inter_ABH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]    

        loss_src = self.autoencoder_src.train_on_batch( [warped_src, target_src_mask], [target_src, target_src_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [warped_dst, target_dst_mask], [target_dst, target_dst_mask] )
        
        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        AA, mAA = self.autoencoder_src.predict([test_A, test_A_m])                                       
        AB, mAB = self.autoencoder_src.predict([test_B, test_B_m])
        BB, mBB = self.autoencoder_dst.predict([test_B, test_B_m])
        
        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                AA[i],
                #mAA[i],
                test_B[i,:,:,0:3], 
                BB[i], 
                #mBB[i],                
                AB[i],
                #mAB[i]
                ), axis=1) )
            
        return [ ('LIAEF128', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        
        face_128_bgr = face[...,0:3]
        face_128_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.autoencoder_src.predict ( [ np.expand_dims(face_128_bgr,0), np.expand_dims(face_128_mask,0) ] )
        x, mx = x[0], mx[0]
        
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        
        if 'erode_mask_modifier' not in in_options.keys():
            in_options['erode_mask_modifier'] = 0
        in_options['erode_mask_modifier'] += 30
            
        if 'blur_mask_modifier' not in in_options.keys():
            in_options['blur_mask_modifier'] = 0
            
        return ConverterMasked(self.predictor_func, predictor_input_size=128, output_size=128, face_type=FaceType.FULL, clip_border_mask_per=0.046875, **in_options)
        
    def Encoder(self, input_layer,):
        x = input_layer
        x = conv(self.keras, x, 128)
        x = conv(self.keras, x, 256)
        x = conv(self.keras, x, 512)
        x = conv(self.keras, x, 1024)
        x = self.keras.layers.Flatten()(x)
        return self.keras.models.Model(input_layer, x)

    def Intermediate(self):
        input_layer = self.keras.layers.Input(shape=(None, 8 * 8 * 1024))
        x = input_layer
        x = self.keras.layers.Dense(256)(x)
        x = self.keras.layers.Dense(8 * 8 * 512)(x)
        x = self.keras.layers.Reshape((8, 8, 512))(x)
        x = upscale(self.keras, x, 512)
        return self.keras.models.Model(input_layer, x)

    def Decoder(self): 
        input_ = self.keras.layers.Input(shape=(16, 16, 1024))
        x = input_
        x = upscale(self.keras, x, 512)
        x = upscale(self.keras, x, 256)
        x = upscale(self.keras, x, 128)
        x = self.keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        
        y = input_  #mask decoder
        y = upscale(self.keras, y, 512)
        y = upscale(self.keras, y, 256)
        y = upscale(self.keras, y, 128)
        y = self.keras.layers.convolutional.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid' )(y)
        
        return self.keras.models.Model(input_, [x,y])
