import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoderMaskH5 = 'decoderMask.h5'
    decoderCommonAH5 = 'decoderCommonA.h5'
    decoderCommonBH5 = 'decoderCommonB.h5'
    decoderRGBH5 = 'decoderRGB.h5'
    decoderBWH5 = 'decoderBW.h5'
    inter_BH5 = 'inter_B.h5'
    inter_AH5 = 'inter_A.h5'

    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {4.5:4,5:4,6:8,7:12,8:16,9:20,10:24,11:24,12:32,13:48} )
                
        ae_input_layer = Input(shape=(128, 128, 3))
        mask_layer = Input(shape=(128, 128, 1)) #same as output

        self.encoder, self.decoderMask, self.decoderCommonA, self.decoderCommonB, self.decoderRGB, \
            self.decoderBW, self.inter_A, self.inter_B = self.Build(ae_input_layer)
            
        if not self.is_first_run():
            self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoderMask.load_weights  (self.get_strpath_storage_for_file(self.decoderMaskH5))
            self.decoderCommonA.load_weights  (self.get_strpath_storage_for_file(self.decoderCommonAH5))
            self.decoderCommonB.load_weights  (self.get_strpath_storage_for_file(self.decoderCommonBH5))
            self.decoderRGB.load_weights  (self.get_strpath_storage_for_file(self.decoderRGBH5))
            self.decoderBW.load_weights  (self.get_strpath_storage_for_file(self.decoderBWH5))
            self.inter_A.load_weights (self.get_strpath_storage_for_file(self.inter_AH5))
            self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))            

        code = self.encoder(ae_input_layer)
        A = self.inter_A(code)
        B = self.inter_B(code)
        
        inter_A_A = Concatenate()([A, A])
        inter_B_A = Concatenate()([B, A])
 
        x1,m1 = self.decoderCommonA (inter_A_A)
        x2,m2 = self.decoderCommonA (inter_A_A)
        self.autoencoder_src     = Model([ae_input_layer,mask_layer],
                    [ self.decoderBW  (Concatenate()([x1,x2]) ),
                      self.decoderMask(Concatenate()([m1,m2]) )
                    ])
                    
        x1,m1 = self.decoderCommonA (inter_A_A)
        x2,m2 = self.decoderCommonB (inter_A_A)
        self.autoencoder_src_RGB = Model([ae_input_layer,mask_layer], 
                    [ self.decoderRGB  (Concatenate()([x1,x2]) ),
                      self.decoderMask (Concatenate()([m1,m2]) )
                    ])
        
        x1,m1 = self.decoderCommonA (inter_B_A)
        x2,m2 = self.decoderCommonB (inter_B_A)
        self.autoencoder_dst     = Model([ae_input_layer,mask_layer], 
                    [ self.decoderRGB  (Concatenate()([x1,x2]) ),
                      self.decoderMask (Concatenate()([m1,m2]) )
                    ])
        
        if self.is_training_mode:
            self.autoencoder_src, self.autoencoder_dst = self.to_multi_gpu_model_if_possible ( [self.autoencoder_src, self.autoencoder_dst] )

        self.autoencoder_src.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMaskLoss([mask_layer]), 'mse'] )
        self.autoencoder_dst.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMaskLoss([mask_layer]), 'mse'] )
  
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(),  batch_size=self.batch_size, 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_GGG, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_G  , 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_GGG, 128] ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(),  batch_size=self.batch_size, 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, 128], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, 128]] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoderMask, self.get_strpath_storage_for_file(self.decoderMaskH5)],
                                [self.decoderCommonA, self.get_strpath_storage_for_file(self.decoderCommonAH5)],
                                [self.decoderCommonB, self.get_strpath_storage_for_file(self.decoderCommonBH5)],
                                [self.decoderRGB, self.get_strpath_storage_for_file(self.decoderRGBH5)],
                                [self.decoderBW, self.get_strpath_storage_for_file(self.decoderBWH5)],
                                [self.inter_A, self.get_strpath_storage_for_file(self.inter_AH5)],
                                [self.inter_B, self.get_strpath_storage_for_file(self.inter_BH5)]] )
        
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask, target_src_GGG = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]    
        
        loss_src = self.autoencoder_src.train_on_batch( [ warped_src, target_src_mask], [ target_src, target_src_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [ warped_dst, target_dst_mask], [ target_dst, target_dst_mask] )
        
        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][3][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]

        AA, mAA = self.autoencoder_src.predict([test_A, test_A_m])                                       
        AB, mAB = self.autoencoder_src_RGB.predict([test_B, test_B_m])
        BB, mBB = self.autoencoder_dst.predict([test_B, test_B_m])
        
        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                np.repeat (np.expand_dims (test_A[i,:,:,0],-1), (3,), -1)  ,
                np.repeat (AA[i], (3,), -1),
                #mAA[i],
                test_B[i,:,:,0:3],
                BB[i], 
                #mBB[i],                
                AB[i],
                #mAB[i]
                ), axis=1) )
            
        return [ ('MIAEF128', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        face_128_bgr = face[...,0:3]
        face_128_mask = np.expand_dims(face[...,-1],-1)
        
        x, mx = self.autoencoder_src_RGB.predict ( [ np.expand_dims(face_128_bgr,0), np.expand_dims(face_128_mask,0) ] )
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

        def DecoderCommon(): 
            input_ = Input(shape=(16, 16, 1024))
            x = input_
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            
            y = input_
            y = upscale(256)(y)
            y = upscale(128)(y)
            y = upscale(64)(y)
            
            return Model(input_, [x,y])
            
        def DecoderRGB(): 
            input_ = Input(shape=(128, 128, 256))
            x = input_
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            return Model(input_, [x])

        def DecoderBW(): 
            input_ = Input(shape=(128, 128, 256))
            x = input_
            x = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(x)
            return Model(input_, [x])
            
        def DecoderMask():
            input_ = Input(shape=(128, 128, 128))        
            y = input_
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
            return Model(input_, [y])
            
        return Encoder(), DecoderMask(), DecoderCommon(), DecoderCommon(), DecoderRGB(), DecoderBW(), Intermediate(), Intermediate()