import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *
from utils.console_utils import *

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'

    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {1.5:2,2:2,3:8,4:16,5:24,6:32,7:40,8:48} )

        bgr_shape, mask_shape, self.encoder, self.decoder_src, self.decoder_dst = self.Build(self.created_vram_gb)
        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))
            
        input_src_bgr = Input(bgr_shape)
        input_src_mask = Input(mask_shape)
        input_dst_bgr = Input(bgr_shape)
        input_dst_mask = Input(mask_shape)

        rec_src_bgr, rec_src_mask = self.decoder_src( self.encoder(input_src_bgr) )        
        rec_dst_bgr, rec_dst_mask = self.decoder_dst( self.encoder(input_dst_bgr) )

        self.ae = Model([input_src_bgr,input_src_mask,input_dst_bgr,input_dst_mask], [rec_src_bgr, rec_src_mask, rec_dst_bgr, rec_dst_mask] )
            
        if self.is_training_mode:
            self.ae, = self.to_multi_gpu_model_if_possible ( [self.ae,] )

        self.ae.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999),
                        loss=[ DSSIMMaskLoss([input_src_mask]), 'mae', DSSIMMaskLoss([input_dst_mask]), 'mae' ] )
  
        self.src_view = K.function([input_src_bgr],[rec_src_bgr, rec_src_mask])
        self.dst_view = K.function([input_dst_bgr],[rec_dst_bgr, rec_dst_mask])
  
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([    
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                            output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_HALF | f.MODE_BGR, 64], 
                                                  [f.TRANSFORMED | f.FACE_ALIGN_HALF | f.MODE_BGR, 64], 
                                                  [f.TRANSFORMED | f.FACE_ALIGN_HALF | f.MODE_M | f.FACE_MASK_FULL, 64] ] ),
                                                  
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_HALF | f.MODE_BGR, 64], 
                                                  [f.TRANSFORMED | f.FACE_ALIGN_HALF | f.MODE_BGR, 64], 
                                                  [f.TRANSFORMED | f.FACE_ALIGN_HALF | f.MODE_M | f.FACE_MASK_FULL, 64] ] )
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

        total, loss_src_bgr, loss_src_mask, loss_dst_bgr, loss_dst_mask = self.ae.train_on_batch( [warped_src, target_src_full_mask, warped_dst, target_dst_full_mask], [target_src, target_src_full_mask, target_dst, target_dst_full_mask] )

        return ( ('loss_src', loss_src_bgr), ('loss_dst', loss_dst_bgr) )
        
    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4]
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        AA, mAA = self.src_view([test_A])                                       
        AB, mAB = self.src_view([test_B])
        BB, mBB = self.dst_view([test_B])
        
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
        
        x, mx = self.src_view ( [ np.expand_dims(face_64_bgr,0) ] )
        x, mx = x[0], mx[0]     
        
        return np.concatenate ( (x,mx), -1 )

    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        return ConverterMasked(self.predictor_func,
                               predictor_input_size=64, 
                               output_size=64, 
                               face_type=FaceType.HALF, 
                               base_erode_mask_modifier=100,
                               base_blur_mask_modifier=100,
                               **in_options)
        
    def Build(self, created_vram_gb):
        exec(nnlib.code_import_all, locals(), globals())
        
        bgr_shape = (64, 64, 3)
        mask_shape = (64, 64, 1)
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return PixelShuffler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func            
        
        def Encoder(input_shape):
            input_layer = Input(input_shape)
            x = input_layer
            if created_vram_gb >= 4:
                x = downscale(128)(x)
                x = downscale(256)(x)
                x = downscale(512)(x)
                x = downscale(1024)(x)
                x = Dense(1024)(Flatten()(x))
                x = Dense(4 * 4 * 1024)(x)
                x = Reshape((4, 4, 1024))(x)
                x = upscale(512)(x)
            else:
                x = downscale(128)(x)
                x = downscale(256)(x)
                x = downscale(512)(x)
                x = downscale(768)(x)
                x = Dense(512)(Flatten()(x))
                x = Dense(4 * 4 * 512)(x)
                x = Reshape((4, 4, 512))(x)
                x = upscale(256)(x)
            return Model(input_layer, x)

        def Decoder():
            if created_vram_gb >= 4:    
                input_ = Input(shape=(8, 8, 512))
                x = input_

                x = upscale(512)(x)
                x = upscale(256)(x)
                x = upscale(128)(x)
                
            else:
                input_ = Input(shape=(8, 8, 256))
                
                x = input_                
                x = upscale(256)(x)
                x = upscale(128)(x)
                x = upscale(64)(x)
                
            y = input_  #mask decoder
            y = upscale(256)(y)
            y = upscale(128)(y)
            y = upscale(64)(y)
            
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
            
            return Model(input_, [x,y])
            
        return bgr_shape, mask_shape, Encoder(bgr_shape), Decoder(), Decoder()