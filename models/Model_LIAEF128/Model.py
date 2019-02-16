import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *
from utils.console_utils import *

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoderH5 = 'decoder.h5'
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'
    
    #override
    def onInitializeOptions(self, is_first_run, ask_override):        
        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = input_bool ("Use pixel loss? (y/n, ?:help skip: n/default ) : ", def_pixel_loss, help_message="Default DSSIM loss good for initial understanding structure of faces. Use pixel loss after 20k epochs to enhance fine details and remove face jitter.")
        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)
            
    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {4.5:4,5:4,6:8,7:12,8:16,9:20,10:24,11:24,12:32,13:48} )

        ae_input_layer = Input(shape=(128, 128, 3))
        mask_layer = Input(shape=(128, 128, 1)) #same as output

        self.encoder, self.decoder, self.inter_B, self.inter_AB = self.Build(ae_input_layer)
       
        if not self.is_first_run():
            self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder.load_weights  (self.get_strpath_storage_for_file(self.decoderH5))
            self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
            self.inter_AB.load_weights (self.get_strpath_storage_for_file(self.inter_ABH5))

        code = self.encoder(ae_input_layer)
        AB = self.inter_AB(code)
        B = self.inter_B(code)
        self.autoencoder_src = Model([ae_input_layer,mask_layer], self.decoder(Concatenate()([AB, AB])) )
        self.autoencoder_dst = Model([ae_input_layer,mask_layer], self.decoder(Concatenate()([B, AB])) )
            
        self.autoencoder_src.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMSEMaskLoss(mask_layer, is_mse=self.options['pixel_loss']), 'mse'] )
        self.autoencoder_dst.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMSEMaskLoss(mask_layer, is_mse=self.options['pixel_loss']), 'mse'] )
  
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([         
                    
            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, 128] ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, 128] ] )
                ])
            
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder, self.get_strpath_storage_for_file(self.decoderH5)],
                                [self.inter_B, self.get_strpath_storage_for_file(self.inter_BH5)],
                                [self.inter_AB, self.get_strpath_storage_for_file(self.inter_ABH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample, generators_list):
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
        return ConverterMasked(self.predictor_func, 
                               predictor_input_size=128, 
                               output_size=128, 
                               face_type=FaceType.FULL, 
                               base_erode_mask_modifier=30,
                               base_blur_mask_modifier=0,
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
            
        def Encoder():
            x = input_layer
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)
            x = Flatten()(x)
            return Model(input_layer, x)

        def Intermediate():
            input_layer = Input(shape=(None, 8 * 8 * 1024))
            x = input_layer
            x = Dense(256)(x)
            x = Dense(8 * 8 * 512)(x)
            x = Reshape((8, 8, 512))(x)
            x = upscale(512)(x)
            return Model(input_layer, x)

        def Decoder(): 
            input_ = Input(shape=(16, 16, 1024))
            x = input_
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            
            y = input_  #mask decoder
            y = upscale(512)(y)
            y = upscale(256)(y)
            y = upscale(128)(y)
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid' )(y)
            
            return Model(input_, [x,y])

        return Encoder(), Decoder(), Intermediate(), Intermediate()