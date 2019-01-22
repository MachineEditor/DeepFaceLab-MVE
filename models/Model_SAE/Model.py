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
    
    decoder_srcH5 = 'decoder_src.h5'
    decoder_srcmH5 = 'decoder_srcm.h5'
    decoder_dstH5 = 'decoder_dst.h5'
    decoder_dstmH5 = 'decoder_dstm.h5'
    
    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_resolution = 128
        default_archi = 'liae'
        default_face_type = 'f'
        
        if is_first_run:
            self.options['resolution'] = input_int("Resolution (64,128 ?:help skip:128) : ", default_resolution, [64,128], help_message="More resolution requires more VRAM.")
            self.options['archi'] = input_str ("AE architecture (df, liae, ?:help skip:%s) : " % (default_archi) , default_archi, ['df','liae'], help_message="DF keeps faces more natural, while LIAE can fix overly different face shapes.").lower()            
            self.options['lighter_encoder'] = input_bool ("Use lightweight encoder? (y/n, ?:help skip:n) : ", False, help_message="Lightweight encoder is 35% faster, requires less VRAM, sacrificing overall quality.")
            self.options['learn_mask'] = input_bool ("Learn mask? (y/n, ?:help skip:y) : ", True, help_message="Choose NO to reduce model size. In this case converter forced to use 'not predicted mask' that is not smooth as predicted. Styled SAE can learn without mask and produce same quality fake if you choose high blur value in converter.")
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['archi'] = self.options.get('archi', default_archi)
            self.options['lighter_encoder'] = self.options.get('lighter_encoder', False)
            self.options['learn_mask'] = self.options.get('learn_mask', True)
            
        default_face_style_power = 10.0
        if is_first_run or ask_override:
            default_face_style_power = default_face_style_power if is_first_run else self.options.get('face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip ( input_number("Face style power ( 0.0 .. 100.0 ?:help skip:%.1f) : " % (default_face_style_power), default_face_style_power, help_message="How fast NN will learn dst face style during generalization of src and dst faces. If style is learned good enough, set this value to 0.01 to prevent artifacts appearing."), 0.0, 100.0 )            
        else:
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
        
        default_bg_style_power = 10.0        
        if is_first_run or ask_override: 
            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power', default_bg_style_power)
            self.options['bg_style_power'] = np.clip ( input_number("Background style power ( 0.0 .. 100.0 ?:help skip:%.1f) : " % (default_bg_style_power), default_bg_style_power, help_message="How fast NN will learn dst background style during generalization of src and dst faces. If style is learned good enough, set this value to 0.1-0.3 to prevent artifacts appearing."), 0.0, 100.0 )            
        else:
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
            
        default_ae_dims = 256 if self.options['archi'] == 'liae' else 512
        default_ed_ch_dims = 42
        if is_first_run:
            self.options['ae_dims'] = np.clip ( input_int("AutoEncoder dims (128-1024 ?:help skip:%d) : " % (default_ae_dims) , default_ae_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 128, 1024 )
            self.options['ed_ch_dims'] = np.clip ( input_int("Encoder/Decoder dims per channel (21-85 ?:help skip:%d) : " % (default_ed_ch_dims) , default_ed_ch_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 21, 85 )
            self.options['face_type'] = input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()            
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['ed_ch_dims'] = self.options.get('ed_ch_dims', default_ed_ch_dims)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
        
        

    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())

        self.set_vram_batch_requirements({2:1,3:2,4:3,5:6,6:8,7:12,8:16})
        
        resolution = self.options['resolution']
        ae_dims = self.options['ae_dims']
        ed_ch_dims = self.options['ed_ch_dims']
        adapt_k_size = False
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)
        
        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)
            
        if self.options['archi'] == 'liae':
            self.encoder = modelify(SAEModel.LIAEEncFlow(resolution, adapt_k_size, self.options['lighter_encoder'], ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))
            
            enc_output_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
            
            self.inter_B = modelify(SAEModel.LIAEInterFlow(resolution, ae_dims=ae_dims)) (enc_output_Inputs)
            self.inter_AB = modelify(SAEModel.LIAEInterFlow(resolution, ae_dims=ae_dims)) (enc_output_Inputs)
            
            inter_output_Inputs = [ Input( np.array(K.int_shape(x)[1:])*(1,1,2) ) for x in self.inter_B.outputs ] 

            self.decoder = modelify(SAEModel.LIAEDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (inter_output_Inputs)
            
            if self.options['learn_mask']:
                self.decoderm = modelify(SAEModel.LIAEDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (inter_output_Inputs)

            if not self.is_first_run():
                self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
                self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
                self.inter_AB.load_weights (self.get_strpath_storage_for_file(self.inter_ABH5))
                self.decoder.load_weights (self.get_strpath_storage_for_file(self.decoderH5))
                if self.options['learn_mask']:
                    self.decoderm.load_weights (self.get_strpath_storage_for_file(self.decodermH5))
     
            warped_src_code = self.encoder (warped_src)
            
            warped_src_inter_AB_code = self.inter_AB (warped_src_code)
            warped_src_inter_code = Concatenate()([warped_src_inter_AB_code,warped_src_inter_AB_code])
            
            pred_src_src = self.decoder(warped_src_inter_code)
            if self.options['learn_mask']:
                pred_src_srcm = self.decoderm(warped_src_inter_code)            
  
            warped_dst_code = self.encoder (warped_dst)
            warped_dst_inter_B_code = self.inter_B (warped_dst_code)
            warped_dst_inter_AB_code = self.inter_AB (warped_dst_code)
            warped_dst_inter_code = Concatenate()([warped_dst_inter_B_code,warped_dst_inter_AB_code])
            pred_dst_dst = self.decoder(warped_dst_inter_code)
            
            if self.options['learn_mask']:
                pred_dst_dstm = self.decoderm(warped_dst_inter_code)
            
            warped_src_dst_inter_code = Concatenate()([warped_dst_inter_AB_code,warped_dst_inter_AB_code])
            pred_src_dst = self.decoder(warped_src_dst_inter_code)
            
            if self.options['learn_mask']:
                pred_src_dstm = self.decoderm(warped_src_dst_inter_code)
        else:
            self.encoder = modelify(SAEModel.DFEncFlow(resolution, adapt_k_size, self.options['lighter_encoder'], ae_dims=ae_dims, ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))

            dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
            
            self.decoder_src = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (dec_Inputs)
            self.decoder_dst = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (dec_Inputs)
            
            if self.options['learn_mask']:
                self.decoder_srcm = modelify(SAEModel.DFDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5))) (dec_Inputs)
                self.decoder_dstm = modelify(SAEModel.DFDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5))) (dec_Inputs)
           
            if not self.is_first_run():
                self.encoder.load_weights      (self.get_strpath_storage_for_file(self.encoderH5))
                self.decoder_src.load_weights  (self.get_strpath_storage_for_file(self.decoder_srcH5))
                self.decoder_dst.load_weights  (self.get_strpath_storage_for_file(self.decoder_dstH5))
                if self.options['learn_mask']:
                    self.decoder_srcm.load_weights (self.get_strpath_storage_for_file(self.decoder_srcmH5))
                    self.decoder_dstm.load_weights (self.get_strpath_storage_for_file(self.decoder_dstmH5))
                
            warped_src_code = self.encoder (warped_src)
            warped_dst_code = self.encoder (warped_dst)
            pred_src_src = self.decoder_src(warped_src_code)
            pred_dst_dst = self.decoder_dst(warped_dst_code)
            pred_src_dst = self.decoder_src(warped_dst_code)
            if self.options['learn_mask']:
                pred_src_srcm = self.decoder_srcm(warped_src_code)
                pred_dst_dstm = self.decoder_dstm(warped_dst_code)
                pred_src_dstm = self.decoder_srcm(warped_dst_code)

        ms_count = len(pred_src_src)
        
        target_src_ar  = [ target_src  if i == 0 else tf.image.resize_bicubic( target_src,  (resolution // (2**i) ,)*2 )  for i in range(ms_count-1, -1, -1)]
        target_srcm_ar = [ target_srcm if i == 0 else tf.image.resize_bicubic( target_srcm, (resolution // (2**i) ,)*2 )  for i in range(ms_count-1, -1, -1)]
        target_dst_ar  = [ target_dst  if i == 0 else tf.image.resize_bicubic( target_dst,  (resolution // (2**i) ,)*2 )  for i in range(ms_count-1, -1, -1)]
        target_dstm_ar = [ target_dstm if i == 0 else tf.image.resize_bicubic( target_dstm, (resolution // (2**i) ,)*2 )  for i in range(ms_count-1, -1, -1)]

        target_srcm_blurred_ar = [ tf_gaussian_blur( max(1, x.get_shape().as_list()[1] // 32) )(x) for x in target_srcm_ar]
        target_srcm_sigm_ar = [ x / 2.0 + 0.5 for x in target_srcm_blurred_ar] 
        target_srcm_anti_sigm_ar = [ 1.0 - x for x in target_srcm_sigm_ar] 
    
        target_dstm_blurred_ar = [ tf_gaussian_blur( max(1, x.get_shape().as_list()[1] // 32) )(x) for x in target_dstm_ar]
        target_dstm_sigm_ar = [ x / 2.0 + 0.5 for x in target_dstm_blurred_ar] 
        target_dstm_anti_sigm_ar = [ 1.0 - x for x in target_dstm_sigm_ar] 
        
        target_src_sigm_ar = [ x + 1 for x in target_src_ar]
        target_dst_sigm_ar = [ x + 1 for x in target_dst_ar]

        pred_src_src_sigm_ar = [ x + 1 for x in pred_src_src]
        pred_dst_dst_sigm_ar = [ x + 1 for x in pred_dst_dst]
        pred_src_dst_sigm_ar = [ x + 1 for x in pred_src_dst]
    
        target_src_masked_ar = [ target_src_sigm_ar[i]*target_srcm_sigm_ar[i]  for i in range(len(target_src_sigm_ar))]
        target_dst_masked_ar = [ target_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(target_dst_sigm_ar))]
        target_dst_anti_masked_ar = [ target_dst_sigm_ar[i]*target_dstm_anti_sigm_ar[i]  for i in range(len(target_dst_sigm_ar))]
  
        psd_target_dst_masked_ar = [ pred_src_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(pred_src_dst_sigm_ar))]
        psd_target_dst_anti_masked_ar = [ pred_src_dst_sigm_ar[i]*target_dstm_anti_sigm_ar[i]  for i in range(len(pred_src_dst_sigm_ar))]
        
        if self.is_training_mode:
            def optimizer():
                return Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
            
            
            if self.options['archi'] == 'liae':
                src_loss_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
                dst_loss_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
                if self.options['learn_mask']:
                    src_mask_loss_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights
                    dst_mask_loss_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights
            else:
                src_loss_train_weights = self.encoder.trainable_weights + self.decoder_src.trainable_weights
                dst_loss_train_weights = self.encoder.trainable_weights + self.decoder_dst.trainable_weights
                if self.options['learn_mask']:
                    src_mask_loss_train_weights = self.encoder.trainable_weights + self.decoder_srcm.trainable_weights
                    dst_mask_loss_train_weights = self.encoder.trainable_weights + self.decoder_dstm.trainable_weights
                
            src_loss = sum([ K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[i], pred_src_src_sigm_ar[i] * target_srcm_sigm_ar[i] ) )) for i in range(len(target_src_masked_ar)) ])
        
            if self.options['face_style_power'] != 0:
                face_style_power = self.options['face_style_power'] / 100.0
                src_loss += tf_style_loss(gaussian_blur_radius=resolution // 8, loss_weight=0.2*face_style_power)( psd_target_dst_masked_ar[-1], target_dst_masked_ar[-1] ) 
                
            if self.options['bg_style_power'] != 0:
                bg_style_power = self.options['bg_style_power'] / 100.0
                src_loss += K.mean( (100*bg_style_power)*K.square(tf_dssim(2.0)( psd_target_dst_anti_masked_ar[-1], target_dst_anti_masked_ar[-1] )))

            self.src_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss], optimizer().get_updates(src_loss, src_loss_train_weights) )
            
            dst_loss = sum([ K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked_ar[i], pred_dst_dst_sigm_ar[i] * target_dstm_sigm_ar[i] ) )) for i in range(len(target_dst_masked_ar)) ])
            self.dst_train = K.function ([warped_dst, target_dst, target_dstm ],[dst_loss], optimizer().get_updates(dst_loss, dst_loss_train_weights) )
   
            if self.options['learn_mask']:
                src_mask_loss = sum([ K.mean(K.square(target_srcm_ar[i]-pred_src_srcm[i])) for i in range(len(target_srcm_ar)) ])
                self.src_mask_train = K.function ([warped_src, target_srcm],[src_mask_loss], optimizer().get_updates(src_mask_loss, src_mask_loss_train_weights) )

                dst_mask_loss = sum([ K.mean(K.square(target_dstm_ar[i]-pred_dst_dstm[i])) for i in range(len(target_dstm_ar)) ])
                self.dst_mask_train = K.function ([warped_dst, target_dstm],[dst_mask_loss], optimizer().get_updates(dst_mask_loss, dst_mask_loss_train_weights) )
            
            self.AE_view = K.function ([warped_src, warped_dst], [pred_src_src[-1], pred_dst_dst[-1], pred_src_dst[-1] ] ) 
            
        else:
            if self.options['learn_mask']:
                self.AE_convert = K.function ([warped_dst],[ pred_src_dst[-1], pred_src_dstm[-1] ])
            else:
                self.AE_convert = K.function ([warped_dst],[ pred_src_dst[-1] ])

        if self.is_training_mode:
            f = SampleProcessor.TypeFlags            
            face_type = f.FACE_ALIGN_FULL if self.options['face_type'] == 'f' else f.FACE_ALIGN_HALF
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],                        
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution]                                              
                                            ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],                                             
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution]                                               
                                            ] )
                ])
    #override
    def onSave(self):
        if self.options['archi'] == 'liae':
            ar = [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                  [self.inter_B, self.get_strpath_storage_for_file(self.inter_BH5)],
                  [self.inter_AB, self.get_strpath_storage_for_file(self.inter_ABH5)],
                  [self.decoder, self.get_strpath_storage_for_file(self.decoderH5)]
                 ]
            if self.options['learn_mask']:
                 ar += [ [self.decoderm, self.get_strpath_storage_for_file(self.decodermH5)] ]
        else:
           ar = [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                 [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                 [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)]
                ]
           if self.options['learn_mask']:
                ar += [ [self.decoder_srcm, self.get_strpath_storage_for_file(self.decoder_srcmH5)],
                        [self.decoder_dstm, self.get_strpath_storage_for_file(self.decoder_dstmH5)] ]
                 
        self.save_weights_safe(ar)
       
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]

        src_loss, = self.src_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
        dst_loss, = self.dst_train ([warped_dst, target_dst, target_dst_mask])
        
        if self.options['learn_mask']:
            src_mask_loss, = self.src_mask_train ([warped_src, target_src_mask])
            dst_mask_loss, = self.dst_mask_train ([warped_dst, target_dst_mask])
        
        return ( ('src_loss', src_loss), ('dst_loss', dst_loss) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]

        S, D, SS, DD, SD, = [ x / 2 + 0.5 for x in ([test_A,test_B] + self.AE_view ([test_A, test_B]) ) ]
        #SM, DM, SDM = [ np.repeat (x, (3,), -1) for x in [SM, DM, SDM] ]

        st_x3 = []
        for i in range(0, len(test_A)):
            st_x3.append ( np.concatenate ( (
                S[i], SS[i], #SM[i],
                D[i], DD[i], #DM[i],
                SD[i], #SDM[i]
                ), axis=1) )
        
        return [ ('SAE', np.concatenate (st_x3, axis=0 )), ]
    
    def predictor_func (self, face):
        face_tanh = face * 2.0 - 1.0        
        face_bgr = face_tanh[...,0:3] 
        prd = [ (x[0] + 1.0) / 2.0 for x in self.AE_convert ( [ np.expand_dims(face_bgr,0) ] ) ]
 
        if not self.options['learn_mask']:
            prd += [ np.expand_dims(face[...,3],-1) ] 
        
        return np.concatenate ( [prd[0], prd[1]], -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked

        base_erode_mask_modifier = 40 if self.options['face_type'] == 'f' else 100
        base_blur_mask_modifier = 10 if self.options['face_type'] == 'f' else 100
        
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF
        
        return ConverterMasked(self.predictor_func, 
                               predictor_input_size=self.options['resolution'], 
                               output_size=self.options['resolution'], 
                               face_type=face_type, 
                               base_erode_mask_modifier=base_erode_mask_modifier,
                               base_blur_mask_modifier=base_blur_mask_modifier,
                               clip_border_mask_per=0.03125,
                               **in_options)
    
    @staticmethod
    def LIAEEncFlow(resolution, adapt_k_size, light_enc, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        
        k_size = resolution // 16 + 1 if adapt_k_size else 5
        strides = resolution // 32 if adapt_k_size else 2
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 

        def downscale_sep (dim):
            def func(x):
                return LeakyReLU(0.1)(SeparableConv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
    
        def func(input):
            ed_dims = K.int_shape(input)[-1]*ed_ch_dims
            
            x = input            
            x = downscale(ed_dims)(x)
            if not light_enc:                
                x = downscale(ed_dims*2)(x)
                x = downscale(ed_dims*4)(x)
                x = downscale(ed_dims*8)(x)
            else:
                x = downscale_sep(ed_dims*2)(x)
                x = downscale(ed_dims*4)(x)
                x = downscale_sep(ed_dims*8)(x)
            
            x = Flatten()(x)               
            return x
        return func
    
    @staticmethod
    def LIAEInterFlow(resolution, ae_dims=256):
        exec (nnlib.import_all(), locals(), globals())
        lowest_dense_res=resolution // 16
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func 
        
        def func(input):   
            x = input[0]
            x = Dense(ae_dims)(x)
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims*2)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims*2))(x)
            x = upscale(ae_dims*2)(x)
            return x
        return func
        
    @staticmethod
    def LIAEDecFlow(output_nc,ed_ch_dims=21,activation='tanh'):
        exec (nnlib.import_all(), locals(), globals())
        ed_dims = output_nc * ed_ch_dims
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func
            
        def to_bgr ():
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding='same', activation='tanh')(x)
            return func
        def func(input):   
            x = input[0]
            x1     = upscale(ed_dims*8)( x )
            x1_bgr = to_bgr()          ( x1 )
            
            x2     = upscale(ed_dims*4)( x1 )
            x2_bgr = to_bgr()          ( x2 )
            
            x3     = upscale(ed_dims*2)( x2 )
            x3_bgr = to_bgr()          ( x3 )
        
            return [ x1_bgr, x2_bgr, x3_bgr ]
        return func
    
    @staticmethod
    def DFEncFlow(resolution, adapt_k_size, light_enc, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        k_size = resolution // 16 + 1 if adapt_k_size else 5
        strides = resolution // 32 if adapt_k_size else 2
        lowest_dense_res = resolution // 16
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def downscale_sep (dim):
            def func(x):
                return LeakyReLU(0.1)(SeparableConv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
    
        def func(input):     
            x = input
            
            ed_dims = K.int_shape(input)[-1]*ed_ch_dims
            
            x = downscale(ed_dims)(x)
            if not light_enc:
                x = downscale(ed_dims*2)(x)
                x = downscale(ed_dims*4)(x)
                x = downscale(ed_dims*8)(x)
            else:
                x = downscale_sep(ed_dims*2)(x)
                x = downscale_sep(ed_dims*4)(x)
                x = downscale_sep(ed_dims*8)(x)
    
            x = Dense(ae_dims)(Flatten()(x))
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
            x = upscale(ae_dims)(x)
               
            return x
        return func
    
    @staticmethod
    def DFDecFlow(output_nc, ed_ch_dims=21):
        exec (nnlib.import_all(), locals(), globals())
        ed_dims = output_nc * ed_ch_dims
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func
            
        def to_bgr ():
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding='same', activation='tanh')(x)
            return func
        def func(input):   
            x = input[0]
            x1     = upscale(ed_dims*8)( x )
            x1_bgr = to_bgr()          ( x1 )
            
            x2     = upscale(ed_dims*4)( x1 )
            x2_bgr = to_bgr()          ( x2 )
            
            x3     = upscale(ed_dims*2)( x2 )
            x3_bgr = to_bgr()          ( x3 )
        
            return [ x1_bgr, x2_bgr, x3_bgr ]
        return func
        
Model = SAEModel