import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {4.5:16,5:16,6:16,7:16,8:24,9:24,10:32,11:32,12:32,13:48} )
                
        ae_input_layer = Input(shape=(128, 128, 3))
        mask_layer = Input(shape=(128, 128, 1)) #same as output
        
        self.encoder, self.decoder_src, self.decoder_dst = self.Build(ae_input_layer)    

        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))

        self.autoencoder_src = Model([ae_input_layer,mask_layer], self.decoder_src(self.encoder(ae_input_layer)))
        self.autoencoder_dst = Model([ae_input_layer,mask_layer], self.decoder_dst(self.encoder(ae_input_layer)))

        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )
                
        self.autoencoder_src.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMaskLoss([mask_layer]), 'mse'] )
        self.autoencoder_dst.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMaskLoss([mask_layer]), 'mse'] )
  
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, 128] ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, 128] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                                [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)]] )
        
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
            
        return [ ('DF', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        
        face_128_bgr = face[...,0:3]
        face_128_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.autoencoder_src.predict ( [ np.expand_dims(face_128_bgr,0), np.expand_dims(face_128_mask,0) ] )
        x, mx = x[0], mx[0]
        
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked            
        return ConverterMasked(self.predictor_func, 
                               predictor_input_size=128, 
                               output_size=128, 
                               face_type=FaceType.FULL, 
                               base_erode_mask_modifier=30,
                               base_blur_mask_modifier=100,
                               **in_options)
        
    def Build(self, input_layer):
        exec(nnlib.code_import_all, locals(), globals())
    
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return PixelShuffler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
            
        def Encoder(input_layer):            
            x = input_layer
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)

            x = Dense(512)(Flatten()(x))
            x = Dense(8 * 8 * 512)(x)
            x = Reshape((8, 8, 512))(x)
            x = upscale(512)(x)
                
            return Model(input_layer, x)

        def Decoder():
            input_ = Input(shape=(16, 16, 512))
            x = input_
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            
            y = input_  #mask decoder
            y = upscale(512)(y)
            y = upscale(256)(y)
            y = upscale(128)(y)
                
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
            
            return Model(input_, [x,y])
        
        return Encoder(input_layer), Decoder(), Decoder()