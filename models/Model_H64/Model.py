from models import ModelBase
from models import TrainingDataType
import numpy as np

from nnlib import DSSIMMaskLossClass
from nnlib import conv
from nnlib import upscale
from facelib import FaceType

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def onInitialize(self, **in_options):
        self.set_vram_batch_requirements( {1.5:2,2:2,3:4,4:8,5:16,6:32,7:32,8:32,9:48} )

        ae_input_layer = self.keras.layers.Input(shape=(64, 64, 3))
        mask_layer = self.keras.layers.Input(shape=(64, 64, 1)) #same as output
      
        self.encoder = self.Encoder(ae_input_layer, self.created_vram_gb)
        self.decoder_src = self.Decoder(self.created_vram_gb)
        self.decoder_dst = self.Decoder(self.created_vram_gb)
        
        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))

        self.autoencoder_src = self.keras.models.Model([ae_input_layer,mask_layer], self.decoder_src(self.encoder(ae_input_layer)))
        self.autoencoder_dst = self.keras.models.Model([ae_input_layer,mask_layer], self.decoder_dst(self.encoder(ae_input_layer)))

        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
        
        optimizer = self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)        
        dssimloss = DSSIMMaskLossClass(self.tf)([mask_layer])
        self.autoencoder_src.compile(optimizer=optimizer, loss=[dssimloss, 'mae'])
        self.autoencoder_dst.compile(optimizer=optimizer, loss=[dssimloss, 'mae'])
  
        if self.is_training_mode:
            from models import TrainingDataGenerator
            f = TrainingDataGenerator.SampleTypeFlags 
            self.set_training_data_generators ([    
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ [f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64], [f.TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64], [f.TRANSFORMED | f.HALF_FACE | f.MODE_M | f.MASK_FULL, 64] ], random_flip=True ),
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ [f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64], [f.TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64], [f.TRANSFORMED | f.HALF_FACE | f.MODE_M | f.MASK_FULL, 64] ], random_flip=True )
                ])
                
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                                [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_full_mask = sample[0]
        warped_dst, target_dst, target_dst_full_mask = sample[1]    
        
        loss_src = self.autoencoder_src.train_on_batch( [warped_src, target_src_full_mask], [target_src, target_src_full_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [warped_dst, target_dst_full_mask], [target_dst, target_dst_full_mask] )

        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )
        
    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4]
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
            
        return [ ('H64', np.concatenate ( st, axis=0 ) ) ]

    def predictor_func (self, face):
        
        face_64_bgr = face[...,0:3]
        face_64_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.autoencoder_src.predict ( [ np.expand_dims(face_64_bgr,0), np.expand_dims(face_64_mask,0) ] )
        x, mx = x[0], mx[0]     
        
        return np.concatenate ( (x,mx), -1 )

    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        
        if 'masked_hist_match' not in in_options.keys() or in_options['masked_hist_match'] is None:
            in_options['masked_hist_match'] = True

        if 'erode_mask_modifier' not in in_options.keys():
            in_options['erode_mask_modifier'] = 0
        in_options['erode_mask_modifier'] += 100
            
        if 'blur_mask_modifier' not in in_options.keys():
            in_options['blur_mask_modifier'] = 0
        in_options['blur_mask_modifier'] += 100
        
        return ConverterMasked(self.predictor_func, predictor_input_size=64, output_size=64, face_type=FaceType.HALF, **in_options)
        
    def Encoder(self, input_layer, created_vram_gb):
        x = input_layer
        if created_vram_gb >= 4:
            x = conv(self.keras, x, 128)
            x = conv(self.keras, x, 256)
            x = conv(self.keras, x, 512)
            x = conv(self.keras, x, 1024)
            x = self.keras.layers.Dense(1024)(self.keras.layers.Flatten()(x))
            x = self.keras.layers.Dense(4 * 4 * 1024)(x)
            x = self.keras.layers.Reshape((4, 4, 1024))(x)
            x = upscale(self.keras, x, 512)
        else:
            x = conv(self.keras, x, 128 )
            x = conv(self.keras, x, 256 )
            x = conv(self.keras, x, 512 )
            x = conv(self.keras, x, 768 )
            x = self.keras.layers.Dense(512)(self.keras.layers.Flatten()(x))
            x = self.keras.layers.Dense(4 * 4 * 512)(x)
            x = self.keras.layers.Reshape((4, 4, 512))(x)
            x = upscale(self.keras, x, 256)
            
        return self.keras.models.Model(input_layer, x)

    def Decoder(self, created_vram_gb):
        if created_vram_gb >= 4:    
            input_ = self.keras.layers.Input(shape=(8, 8, 512))
        else:
            input_ = self.keras.layers.Input(shape=(8, 8, 256))
            
        x = input_
        x = upscale(self.keras, x, 256)
        x = upscale(self.keras, x, 128)
        x = upscale(self.keras, x, 64)
        
        y = input_  #mask decoder
        y = upscale(self.keras, y, 256)
        y = upscale(self.keras, y, 128)
        y = upscale(self.keras, y, 64)
        
        x = self.keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        y = self.keras.layers.convolutional.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
        
        
        return self.keras.models.Model(input_, [x,y])
