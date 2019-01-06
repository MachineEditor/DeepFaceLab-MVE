import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *
from utils.console_utils import *

#SAE - Styled AutoEncoder
class SAEModel(ModelBase):

    encoderH5 = 'encoder.h5'    
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'
    decoderH5 = 'decoder.h5'
    decodermH5 = 'decoderm.h5'
    
    #override
    def onInitializeOptions(self, is_first_run, ask_for_session_options):
        default_resolution = 128
        default_ae_dims = 256
        default_face_type = 'f'
        
        if is_first_run: 
            #first run
            self.options['resolution'] = input_int("Resolution (valid: 64,128, skip:128) : ", default_resolution, [64,128])
            self.options['ae_dims'] = input_int("AutoEncoder dims (valid: 128,256,512 skip:256) : ", default_ae_dims, [128,256,512])
            self.options['face_type'] = input_str ("Half or Full face? (h/f, skip:f) : ", default_face_type, ['h','f'])
            
        else: 
            #not first run
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
    
    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())

        self.set_vram_batch_requirements({2:2,3:3,4:4,5:4,6:8,7:12,8:16})
        
        resolution = self.options['resolution']
        ae_dims = self.options['ae_dims']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        self.encoder = modelify(SAEModel.EncFlow()  ) (Input(bgr_shape))
        
        enc_output_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
        
        self.inter_B = modelify(SAEModel.InterFlow(dims=ae_dims,lowest_dense_res=resolution // 16)) (enc_output_Inputs)
        self.inter_AB = modelify(SAEModel.InterFlow(dims=ae_dims,lowest_dense_res=resolution // 16)) (enc_output_Inputs)
        
        inter_output_Inputs = [ Input( np.array(K.int_shape(x)[1:])*(1,1,2) ) for x in self.inter_B.outputs ] 

        self.decoder = modelify(SAEModel.DecFlow (bgr_shape[2],dims=ae_dims*2)) (inter_output_Inputs)
        self.decoderm = modelify(SAEModel.DecFlow (mask_shape[2],dims=ae_dims)) (inter_output_Inputs)
        
        if not self.is_first_run():
            self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
            self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
            self.inter_AB.load_weights (self.get_strpath_storage_for_file(self.inter_ABH5))
            self.decoder.load_weights (self.get_strpath_storage_for_file(self.decoderH5))
            self.decoderm.load_weights (self.get_strpath_storage_for_file(self.decodermH5))
 
        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)
        
        warped_src_code = self.encoder (warped_src)
        
        warped_src_inter_AB_code = self.inter_AB (warped_src_code)
        warped_src_inter_code = Concatenate()([warped_src_inter_AB_code,warped_src_inter_AB_code])
        
        pred_src_src = self.decoder(warped_src_inter_code)
        pred_src_srcm = self.decoderm(warped_src_inter_code)
        
        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)
        
        warped_dst_code = self.encoder (warped_dst)
        warped_dst_inter_B_code = self.inter_B (warped_dst_code)
        warped_dst_inter_AB_code = self.inter_AB (warped_dst_code)
        warped_dst_inter_code = Concatenate()([warped_dst_inter_B_code,warped_dst_inter_AB_code])
        pred_dst_dst = self.decoder(warped_dst_inter_code)
        pred_dst_dstm = self.decoderm(warped_dst_inter_code)
        
        warped_src_dst_inter_code = Concatenate()([warped_dst_inter_AB_code,warped_dst_inter_AB_code])
        pred_src_dst = self.decoder(warped_src_dst_inter_code)
        pred_src_dstm = self.decoderm(warped_src_dst_inter_code)
        
        target_srcm_blurred = tf_gaussian_blur(resolution // 32)(target_srcm)        
        target_srcm_sigm = target_srcm_blurred / 2.0 + 0.5
        target_srcm_anti_sigm = 1.0 - target_srcm_sigm

        target_dstm_blurred = tf_gaussian_blur(resolution // 32)(target_dstm)
        target_dstm_sigm = target_dstm_blurred / 2.0 + 0.5
        target_dstm_anti_sigm = 1.0 - target_dstm_sigm
        
        target_src_sigm = target_src+1
        target_dst_sigm = target_dst+1
        
        pred_src_src_sigm = pred_src_src+1
        pred_dst_dst_sigm = pred_dst_dst+1
        pred_src_dst_sigm = pred_src_dst+1

        pred_src_dstm_blurred = tf_gaussian_blur(resolution // 32)(pred_src_dstm)
        pred_src_dstm_sigm = pred_src_dstm_blurred / 2.0 + 0.5
        pred_src_dstm_anti_sigm = 1.0 - pred_src_dstm_sigm
        
        target_src_masked = target_src_sigm*target_srcm_sigm
        
        target_dst_masked = target_dst_sigm * target_dstm_sigm
        target_dst_anti_masked = target_dst_sigm * target_dstm_anti_sigm
        
        target_dst_psd_masked = target_dst_sigm * pred_src_dstm_sigm
        target_dst_psd_anti_masked = target_dst_sigm * pred_src_dstm_anti_sigm
        
        pred_src_src_masked = pred_src_src_sigm * target_srcm_sigm        
        pred_dst_dst_masked = pred_dst_dst_sigm * target_dstm_sigm
        
        psd_target_dst_masked = pred_src_dst_sigm * target_dstm_sigm
        psd_target_dst_anti_masked = pred_src_dst_sigm * target_dstm_anti_sigm
        
        psd_psd_masked = pred_src_dst_sigm * pred_src_dstm_sigm
        psd_psd_anti_masked = pred_src_dst_sigm * pred_src_dstm_anti_sigm

        src_loss = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked, pred_src_src_masked )) ) 
        src_loss += tf_style_loss(gaussian_blur_radius=resolution // 8, loss_weight=0.2)(psd_target_dst_masked, target_dst_masked) 
        src_loss += K.mean( 100*K.square(tf_dssim(2.0)( psd_target_dst_anti_masked, target_dst_anti_masked )))
        #src_loss += tf_style_loss(gaussian_blur_radius=resolution // 8, loss_weight=0.2)(psd_psd_masked, target_dst_psd_masked) 
        #src_loss += K.mean( 100*K.square(tf_dssim(2.0)( psd_psd_anti_masked, target_dst_psd_anti_masked )))

        self.src_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(src_loss, self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights) )
                                    
        dst_loss = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked, pred_dst_dst_masked )) )        
        self.dst_train = K.function ([warped_dst, target_dst, target_dstm],[dst_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(dst_loss, self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights) )
                                    
        
        src_mask_loss = K.mean(K.square(target_srcm-pred_src_srcm))                                       
        self.src_mask_train = K.function ([warped_src, target_srcm],[src_mask_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(src_mask_loss, self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights) )
        
        dst_mask_loss = K.mean(K.square(target_dstm-pred_dst_dstm))                                       
        self.dst_mask_train = K.function ([warped_dst, target_dstm],[dst_mask_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(dst_mask_loss, self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights) )
                                    
        self.AE_view = K.function ([warped_src, warped_dst],[pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm])
        self.AE_convert = K.function ([warped_dst],[pred_src_dst, pred_src_dstm])
        
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            
            face_type = f.FACE_ALIGN_FULL if self.options['face_type'] == 'f' else f.FACE_ALIGN_HALF
            
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution],
                                              #
                                              
                                              ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                 [self.inter_B, self.get_strpath_storage_for_file(self.inter_BH5)],
                                 [self.inter_AB, self.get_strpath_storage_for_file(self.inter_ABH5)],
                                 [self.decoder, self.get_strpath_storage_for_file(self.decoderH5)],
                                 [self.decoderm, self.get_strpath_storage_for_file(self.decodermH5)],
                                ] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]    

        src_loss, = self.src_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
        dst_loss, = self.dst_train ([warped_dst, target_dst, target_dst_mask])
        
        src_mask_loss, = self.src_mask_train ([warped_src, target_src_mask])
        dst_mask_loss, = self.dst_mask_train ([warped_dst, target_dst_mask])
        
        return ( ('src_loss', src_loss), ('dst_loss', dst_loss) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        S = test_A
        D = test_B
        
        SS, SM, DD, DM, SD, SDM = self.AE_view ([test_A, test_B])        
        S, D, SS, SM, DD, DM, SD, SDM = [ x / 2 + 0.5 for x in [S, D, SS, SM, DD, DM, SD, SDM] ]

        SM, DM, SDM = [ np.repeat (x, (3,), -1) for x in [SM, DM, SDM] ]
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                S[i], SS[i], #SM[i],
                D[i], DD[i], #DM[i],
                SD[i], #SDM[i]
                ), axis=1) )
            
        return [ ('SAE', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):        
        face = face * 2.0 - 1.0        
        face_128_bgr = face[...,0:3] 
        x, mx = [ (x[0] + 1.0) / 2.0 for x in self.AE_convert ( [ np.expand_dims(face_128_bgr,0) ] ) ]
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked

        base_erode_mask_modifier = 30 if self.options['face_type'] == 'f' else 100
        base_blur_mask_modifier = 0 if self.options['face_type'] == 'f' else 100
        
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF
        
        return ConverterMasked(self.predictor_func, 
                               predictor_input_size=self.options['resolution'], 
                               output_size=self.options['resolution'], 
                               face_type=face_type, 
                               base_erode_mask_modifier=base_erode_mask_modifier,
                               base_blur_mask_modifier=base_blur_mask_modifier,
                               **in_options)
    
    @staticmethod
    def EncFlow():
        exec (nnlib.import_all(), locals(), globals())
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
    
        def func(input):     
            x = input
            
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)    
            x = Flatten()(x)               
            return x
        return func
    
    @staticmethod
    def InterFlow(dims=256, lowest_dense_res=8):
        exec (nnlib.import_all(), locals(), globals())
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func 
        
        def func(input):   
            x = input[0]
            x = Dense(dims)(x)
            x = Dense(lowest_dense_res * lowest_dense_res * dims*2)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, dims*2))(x)
            x = upscale(dims*2)(x)
            return x
        return func
        
    @staticmethod
    def DecFlow(output_nc,dims,activation='tanh'):
        exec (nnlib.import_all(), locals(), globals())
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
            
        def func(input):
            x = input[0]
            x = upscale(dims)(x)
            x = upscale(dims//2)(x)
            x = upscale(dims//4)(x)
                
            x = Conv2D(output_nc, kernel_size=5, padding='same', activation=activation)(x)
            return x
            
        return func

Model = SAEModel