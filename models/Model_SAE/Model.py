import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *
from interact import interact as io

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
        yn_str = {True:'y',False:'n'}
        
        default_resolution = 128
        default_archi = 'df'
        default_face_type = 'f'
        
        if is_first_run:
            resolution = io.input_int("Resolution ( 64-256 ?:help skip:128) : ", default_resolution, help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
            resolution = np.clip (resolution, 64, 256)            
            while np.modf(resolution / 16)[0] != 0.0:
                resolution -= 1
            self.options['resolution'] = resolution
            
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()            
            self.options['learn_mask'] = io.input_bool ("Learn mask? (y/n, ?:help skip:y) : ", True, help_message="Learning mask can help model to recognize face directions. Learn without mask can reduce model size, in this case converter forced to use 'not predicted mask' that is not smooth as predicted. Model with style values can be learned without mask and produce same quality result.")
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            self.options['learn_mask'] = self.options.get('learn_mask', True)
            
            
        if is_first_run and 'tensorflow' in self.device_config.backend:
            def_optimizer_mode = self.options.get('optimizer_mode', 1)
            self.options['optimizer_mode'] = io.input_int ("Optimizer mode? ( 1,2,3 ?:help skip:%d) : " % (def_optimizer_mode), def_optimizer_mode, help_message="1 - no changes. 2 - allows you to train x2 bigger network consuming RAM. 3 - allows you to train x3 bigger network consuming huge amount of RAM and slower, depends on CPU power.")
        else:
            self.options['optimizer_mode'] = self.options.get('optimizer_mode', 1)
        
        if is_first_run:
            self.options['archi'] = io.input_str ("AE architecture (df, liae, vg ?:help skip:%s) : " % (default_archi) , default_archi, ['df','liae','vg'], help_message="'df' keeps faces more natural. 'liae' can fix overly different face shapes. 'vg' - currently testing.").lower()
        else:
            self.options['archi'] = self.options.get('archi', default_archi)
        
        default_ae_dims = 256 if self.options['archi'] == 'liae' else 512
        default_ed_ch_dims = 42
        def_ca_weights = False
        if is_first_run:
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dims (32-1024 ?:help skip:%d) : " % (default_ae_dims) , default_ae_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )
            self.options['ed_ch_dims'] = np.clip ( io.input_int("Encoder/Decoder dims per channel (21-85 ?:help skip:%d) : " % (default_ed_ch_dims) , default_ed_ch_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 21, 85 )
            self.options['ca_weights'] = io.input_bool ("Use CA weights? (y/n, ?:help skip: %s ) : " % (yn_str[def_ca_weights]), def_ca_weights, help_message="Initialize network with 'Convolution Aware' weights. This may help to achieve a higher accuracy model, but consumes time at first run.")
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['ed_ch_dims'] = self.options.get('ed_ch_dims', default_ed_ch_dims)
            self.options['ca_weights'] = self.options.get('ca_weights', def_ca_weights)
            
        if is_first_run:
            self.options['lighter_encoder'] = io.input_bool ("Use lightweight encoder? (y/n, ?:help skip:n) : ", False, help_message="Lightweight encoder is 35% faster, requires less VRAM, but sacrificing overall quality.")
            
            if self.options['archi'] != 'vg':
                self.options['multiscale_decoder'] = io.input_bool ("Use multiscale decoder? (y/n, ?:help skip:n) : ", False, help_message="Multiscale decoder helps to get better details.")
        else:
            self.options['lighter_encoder'] = self.options.get('lighter_encoder', False)
            
            if self.options['archi'] != 'vg':
                self.options['multiscale_decoder'] = self.options.get('multiscale_decoder', False)
            
        default_face_style_power = 0.0        
        default_bg_style_power = 0.0  
        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = io.input_bool ("Use pixel loss? (y/n, ?:help skip: %s ) : " % (yn_str[def_pixel_loss]), def_pixel_loss, help_message="Default DSSIM loss good for initial understanding structure of faces. Use pixel loss after 15-25k iters to enhance fine details and decrease face jitter.")
        
            default_face_style_power = default_face_style_power if is_first_run else self.options.get('face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip ( io.input_number("Face style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_face_style_power), default_face_style_power, 
                                                                               help_message="Learn to transfer face style details such as light and color conditions. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.1 value and check history changes."), 0.0, 100.0 )            
                            
            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power', default_bg_style_power)
            self.options['bg_style_power'] = np.clip ( io.input_number("Background style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_bg_style_power), default_bg_style_power, 
                                                                               help_message="Learn to transfer image around face. This can make face more like dst."), 0.0, 100.0 )            
        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)

    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        SAEModel.initialize_nn_functions()
        self.set_vram_batch_requirements({1.5:4})
        
        resolution = self.options['resolution']
        ae_dims = self.options['ae_dims']
        ed_ch_dims = self.options['ed_ch_dims']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        self.ms_count = ms_count = 3 if (self.options['archi'] != 'vg' and self.options['multiscale_decoder']) else 1
        
        masked_training = True
        
        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)
        
        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)

        target_src_ar = [ Input ( ( bgr_shape[0] // (2**i) ,)*2 + (bgr_shape[-1],) ) for i in range(ms_count-1, -1, -1)]
        target_srcm_ar = [ Input ( ( mask_shape[0] // (2**i) ,)*2 + (mask_shape[-1],) ) for i in range(ms_count-1, -1, -1)]
        target_dst_ar  = [ Input ( ( bgr_shape[0] // (2**i) ,)*2 + (bgr_shape[-1],) ) for i in range(ms_count-1, -1, -1)]
        target_dstm_ar = [ Input ( ( mask_shape[0] // (2**i) ,)*2 + (mask_shape[-1],) ) for i in range(ms_count-1, -1, -1)]

    
        models_list = []
        weights_to_load = []
        if self.options['archi'] == 'liae':
            self.encoder = modelify(SAEModel.LIAEEncFlow(resolution, self.options['lighter_encoder'], ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))
            
            enc_output_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
            
            self.inter_B = modelify(SAEModel.LIAEInterFlow(resolution, ae_dims=ae_dims)) (enc_output_Inputs)
            self.inter_AB = modelify(SAEModel.LIAEInterFlow(resolution, ae_dims=ae_dims)) (enc_output_Inputs)
            
            inter_output_Inputs = [ Input( np.array(K.int_shape(x)[1:])*(1,1,2) ) for x in self.inter_B.outputs ] 

            self.decoder = modelify(SAEModel.LIAEDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2, multiscale_count=self.ms_count )) (inter_output_Inputs)
            
            models_list += [self.encoder, self.inter_B, self.inter_AB, self.decoder]
            
            if self.options['learn_mask']:
                self.decoderm = modelify(SAEModel.LIAEDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (inter_output_Inputs)
                models_list += [self.decoderm]
                
            if not self.is_first_run():
                weights_to_load += [  [self.encoder , 'encoder.h5'],
                                      [self.inter_B , 'inter_B.h5'],
                                      [self.inter_AB, 'inter_AB.h5'],
                                      [self.decoder , 'decoder.h5'],
                                    ]
                if self.options['learn_mask']:
                    weights_to_load += [ [self.decoderm, 'decoderm.h5'] ]
            
            warped_src_code = self.encoder (warped_src)            
            warped_src_inter_AB_code = self.inter_AB (warped_src_code)
            warped_src_inter_code = Concatenate()([warped_src_inter_AB_code,warped_src_inter_AB_code])            
            
            warped_dst_code = self.encoder (warped_dst)
            warped_dst_inter_B_code = self.inter_B (warped_dst_code)
            warped_dst_inter_AB_code = self.inter_AB (warped_dst_code)
            warped_dst_inter_code = Concatenate()([warped_dst_inter_B_code,warped_dst_inter_AB_code])

            warped_src_dst_inter_code = Concatenate()([warped_dst_inter_AB_code,warped_dst_inter_AB_code])
            
            pred_src_src = self.decoder(warped_src_inter_code)
            pred_dst_dst = self.decoder(warped_dst_inter_code)            
            pred_src_dst = self.decoder(warped_src_dst_inter_code)
            
            if self.options['learn_mask']:
                pred_src_srcm = self.decoderm(warped_src_inter_code)
                pred_dst_dstm = self.decoderm(warped_dst_inter_code)
                pred_src_dstm = self.decoderm(warped_src_dst_inter_code)

        elif self.options['archi'] == 'df':
            self.encoder = modelify(SAEModel.DFEncFlow(resolution, self.options['lighter_encoder'], ae_dims=ae_dims, ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))

            dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
            
            self.decoder_src = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2, multiscale_count=self.ms_count )) (dec_Inputs)
            self.decoder_dst = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2, multiscale_count=self.ms_count )) (dec_Inputs)
            
            models_list += [self.encoder, self.decoder_src, self.decoder_dst]
            
            if self.options['learn_mask']:
                self.decoder_srcm = modelify(SAEModel.DFDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (dec_Inputs)
                self.decoder_dstm = modelify(SAEModel.DFDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (dec_Inputs)
                models_list += [self.decoder_srcm, self.decoder_dstm]
                
            if not self.is_first_run():
                weights_to_load += [  [self.encoder    , 'encoder.h5'],
                                      [self.decoder_src, 'decoder_src.h5'],
                                      [self.decoder_dst, 'decoder_dst.h5']
                                    ]
                if self.options['learn_mask']:
                    weights_to_load += [ [self.decoder_srcm, 'decoder_srcm.h5'],
                                         [self.decoder_dstm, 'decoder_dstm.h5'],
                                       ]
            
            
                
                
                
            warped_src_code = self.encoder (warped_src)
            warped_dst_code = self.encoder (warped_dst)
            pred_src_src = self.decoder_src(warped_src_code)
            pred_dst_dst = self.decoder_dst(warped_dst_code)
            pred_src_dst = self.decoder_src(warped_dst_code)
            
            if self.options['learn_mask']:
                pred_src_srcm = self.decoder_srcm(warped_src_code)
                pred_dst_dstm = self.decoder_dstm(warped_dst_code)
                pred_src_dstm = self.decoder_srcm(warped_dst_code)
                
        elif self.options['archi'] == 'vg':
            self.encoder = modelify(SAEModel.VGEncFlow(resolution, self.options['lighter_encoder'], ae_dims=ae_dims, ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))

            dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
            
            self.decoder_src = modelify(SAEModel.VGDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2 )) (dec_Inputs)
            self.decoder_dst = modelify(SAEModel.VGDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2 )) (dec_Inputs)
            
            models_list += [self.encoder, self.decoder_src, self.decoder_dst]
            
            if self.options['learn_mask']:
                self.decoder_srcm = modelify(SAEModel.VGDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (dec_Inputs)
                self.decoder_dstm = modelify(SAEModel.VGDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (dec_Inputs)
                models_list += [self.decoder_srcm, self.decoder_dstm]
                
            if not self.is_first_run():
                weights_to_load += [  [self.encoder    , 'encoder.h5'],
                                      [self.decoder_src, 'decoder_src.h5'],
                                      [self.decoder_dst, 'decoder_dst.h5']
                                    ]
                if self.options['learn_mask']:
                    weights_to_load += [ [self.decoder_srcm, 'decoder_srcm.h5'],
                                         [self.decoder_dstm, 'decoder_dstm.h5'],
                                       ]
            
            warped_src_code = self.encoder (warped_src)
            warped_dst_code = self.encoder (warped_dst)
            pred_src_src = self.decoder_src(warped_src_code)
            pred_dst_dst = self.decoder_dst(warped_dst_code)
            pred_src_dst = self.decoder_src(warped_dst_code)


            if self.options['learn_mask']:
                pred_src_srcm = self.decoder_srcm(warped_src_code)
                pred_dst_dstm = self.decoder_dstm(warped_dst_code)
                pred_src_dstm = self.decoder_srcm(warped_dst_code)
                
        if self.is_first_run() and self.options['ca_weights']:
            io.log_info ("Initializing CA weights...")
            conv_weights_list = []
            for model in models_list:
                for layer in model.layers:
                    if type(layer) == Conv2D:
                        conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights            
            CAInitializerMP ( conv_weights_list )
            
                
        pred_src_src, pred_dst_dst, pred_src_dst, = [ [x] if type(x) != list else x for x in [pred_src_src, pred_dst_dst, pred_src_dst, ] ]
        
        if self.options['learn_mask']:
            pred_src_srcm, pred_dst_dstm, pred_src_dstm = [ [x] if type(x) != list else x for x in [pred_src_srcm, pred_dst_dstm, pred_src_dstm] ]

        target_srcm_blurred_ar = [ gaussian_blur( max(1, K.int_shape(x)[1] // 32) )(x) for x in target_srcm_ar]
        target_srcm_sigm_ar = target_srcm_blurred_ar #[ x / 2.0 + 0.5 for x in target_srcm_blurred_ar] 
        target_srcm_anti_sigm_ar = [ 1.0 - x for x in target_srcm_sigm_ar] 
    
        target_dstm_blurred_ar = [ gaussian_blur( max(1, K.int_shape(x)[1] // 32) )(x) for x in target_dstm_ar]
        target_dstm_sigm_ar = target_dstm_blurred_ar#[ x / 2.0 + 0.5 for x in target_dstm_blurred_ar] 
        target_dstm_anti_sigm_ar = [ 1.0 - x for x in target_dstm_sigm_ar] 
        
        target_src_sigm_ar = target_src_ar#[ x + 1 for x in target_src_ar]
        target_dst_sigm_ar = target_dst_ar#[ x + 1 for x in target_dst_ar]

        pred_src_src_sigm_ar = pred_src_src#[ x + 1 for x in pred_src_src]
        pred_dst_dst_sigm_ar = pred_dst_dst#[ x + 1 for x in pred_dst_dst]
        pred_src_dst_sigm_ar = pred_src_dst#[ x + 1 for x in pred_src_dst]

        target_src_masked_ar = [ target_src_sigm_ar[i]*target_srcm_sigm_ar[i]  for i in range(len(target_src_sigm_ar))]
        target_dst_masked_ar = [ target_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(target_dst_sigm_ar))]
        target_dst_anti_masked_ar = [ target_dst_sigm_ar[i]*target_dstm_anti_sigm_ar[i]  for i in range(len(target_dst_sigm_ar))]
    
        pred_src_src_masked_ar = [ pred_src_src_sigm_ar[i] * target_srcm_sigm_ar[i]  for i in range(len(pred_src_src_sigm_ar))]
        pred_dst_dst_masked_ar = [ pred_dst_dst_sigm_ar[i] * target_dstm_sigm_ar[i]  for i in range(len(pred_dst_dst_sigm_ar))]
        
        target_src_masked_ar_opt = target_src_masked_ar if masked_training else target_src_sigm_ar
        target_dst_masked_ar_opt = target_dst_masked_ar if masked_training else target_dst_sigm_ar
        
        pred_src_src_masked_ar_opt = pred_src_src_masked_ar if masked_training else pred_src_src_sigm_ar
        pred_dst_dst_masked_ar_opt = pred_dst_dst_masked_ar if masked_training else pred_dst_dst_sigm_ar
        
        psd_target_dst_masked_ar = [ pred_src_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(pred_src_dst_sigm_ar))]
        psd_target_dst_anti_masked_ar = [ pred_src_dst_sigm_ar[i]*target_dstm_anti_sigm_ar[i]  for i in range(len(pred_src_dst_sigm_ar))]
        
        if self.is_training_mode:            
            self.src_dst_opt      = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999, tf_cpu_mode=self.options['optimizer_mode']-1)
            self.src_dst_mask_opt = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999, tf_cpu_mode=self.options['optimizer_mode']-1)

            if self.options['archi'] == 'liae':          
                src_dst_loss_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
                if self.options['learn_mask']:
                    src_dst_mask_loss_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights
            else:   
                src_dst_loss_train_weights = self.encoder.trainable_weights + self.decoder_src.trainable_weights + self.decoder_dst.trainable_weights
                if self.options['learn_mask']:
                    src_dst_mask_loss_train_weights = self.encoder.trainable_weights + self.decoder_srcm.trainable_weights + self.decoder_dstm.trainable_weights
            
            if not self.options['pixel_loss']:
                src_loss_batch = sum([ (  100*K.square( dssim(kernel_size=int(resolution/11.6),max_value=1.0)( target_src_masked_ar_opt[i],  pred_src_src_masked_ar_opt[i] ) )) for i in range(len(target_src_masked_ar_opt)) ])
            else:
                src_loss_batch = sum([ K.mean ( 100*K.square( target_src_masked_ar_opt[i] - pred_src_src_masked_ar_opt[i] ), axis=[1,2,3]) for i in range(len(target_src_masked_ar_opt)) ])

            src_loss = K.mean(src_loss_batch)

            face_style_power = self.options['face_style_power']  / 100.0
            
            if face_style_power != 0:    
                src_loss += style_loss(gaussian_blur_radius=resolution//16, loss_weight=face_style_power, wnd_size=0)( psd_target_dst_masked_ar[-1], target_dst_masked_ar[-1] ) 

            bg_style_power = self.options['bg_style_power'] / 100.0
            if bg_style_power != 0:
                if not self.options['pixel_loss']:
                    bg_loss = K.mean( (100*bg_style_power)*K.square(dssim(kernel_size=int(resolution/11.6),max_value=1.0)( psd_target_dst_anti_masked_ar[-1], target_dst_anti_masked_ar[-1] )))
                else:
                    bg_loss = K.mean( (100*bg_style_power)*K.square( psd_target_dst_anti_masked_ar[-1] - target_dst_anti_masked_ar[-1] ))
                src_loss += bg_loss

            if not self.options['pixel_loss']:
                dst_loss_batch = sum([ (  100*K.square(dssim(kernel_size=int(resolution/11.6),max_value=1.0)( target_dst_masked_ar_opt[i],  pred_dst_dst_masked_ar_opt[i] ) )) for i in range(len(target_dst_masked_ar_opt)) ])
            else:
                dst_loss_batch = sum([ K.mean ( 100*K.square( target_dst_masked_ar_opt[i] - pred_dst_dst_masked_ar_opt[i] ), axis=[1,2,3]) for i in range(len(target_dst_masked_ar_opt)) ])
                
            dst_loss = K.mean(dst_loss_batch)

            feed = [warped_src, warped_dst]            
            feed += target_src_ar[::-1]
            feed += target_srcm_ar[::-1]
            feed += target_dst_ar[::-1]
            feed += target_dstm_ar[::-1]
    
            self.src_dst_train = K.function (feed,[src_loss,dst_loss], self.src_dst_opt.get_updates(src_loss+dst_loss, src_dst_loss_train_weights) )

            if self.options['learn_mask']:
                src_mask_loss = sum([ K.mean(K.square(target_srcm_ar[-1]-pred_src_srcm[-1])) for i in range(len(target_srcm_ar)) ])
                dst_mask_loss = sum([ K.mean(K.square(target_dstm_ar[-1]-pred_dst_dstm[-1])) for i in range(len(target_dstm_ar)) ])
                
                feed = [ warped_src, warped_dst]    
                feed += target_srcm_ar[::-1]
                feed += target_dstm_ar[::-1]            
                
                self.src_dst_mask_train = K.function (feed,[src_mask_loss, dst_mask_loss], self.src_dst_mask_opt.get_updates(src_mask_loss+dst_mask_loss, src_dst_mask_loss_train_weights) )

            if self.options['learn_mask']:
                self.AE_view = K.function ([warped_src, warped_dst], [pred_src_src[-1], pred_dst_dst[-1], pred_src_dst[-1], pred_src_dstm[-1]])
            else:
                self.AE_view = K.function ([warped_src, warped_dst], [pred_src_src[-1], pred_dst_dst[-1], pred_src_dst[-1] ] ) 
                
            self.load_weights_safe(weights_to_load)#, [ [self.src_dst_opt, 'src_dst_opt'], [self.src_dst_mask_opt, 'src_dst_mask_opt']])
        else:
            self.load_weights_safe(weights_to_load)
            if self.options['learn_mask']:
                self.AE_convert = K.function ([warped_dst],[ pred_src_dst[-1], pred_src_dstm[-1] ])
            else:
                self.AE_convert = K.function ([warped_dst],[ pred_src_dst[-1] ])
                
        
        if self.is_training_mode:            
            self.src_sample_losses = []
            self.dst_sample_losses = []
            
            f = SampleProcessor.TypeFlags            
            face_type = f.FACE_ALIGN_FULL if self.options['face_type'] == 'f' else f.FACE_ALIGN_HALF
            
            output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution] ]            
            output_sample_types += [ [f.TRANSFORMED | face_type | f.MODE_BGR, resolution // (2**i) ] for i in range(ms_count)]
            output_sample_types += [ [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution // (2**i) ] for i in range(ms_count)]
            
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ), 
                        output_sample_types=output_sample_types ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, ), 
                        output_sample_types=output_sample_types )
                ])
            
    #override
    def onSave(self):
        opt_ar = [ [self.src_dst_opt,      'src_dst_opt'],
                   [self.src_dst_mask_opt, 'src_dst_mask_opt']
                 ]
        ar = []
        if self.options['archi'] == 'liae':
            ar += [[self.encoder, 'encoder.h5'],
                  [self.inter_B, 'inter_B.h5'],
                  [self.inter_AB, 'inter_AB.h5'],
                  [self.decoder, 'decoder.h5']
                 ]
            if self.options['learn_mask']:
                 ar += [ [self.decoderm, 'decoderm.h5'] ]
        elif self.options['archi'] == 'df' or self.options['archi'] == 'vg':
           ar += [[self.encoder, 'encoder.h5'],
                 [self.decoder_src, 'decoder_src.h5'],
                 [self.decoder_dst, 'decoder_dst.h5']
                ]
           if self.options['learn_mask']:
                ar += [ [self.decoder_srcm, 'decoder_srcm.h5'],
                        [self.decoder_dstm, 'decoder_dstm.h5'] ]
               
        self.save_weights_safe(ar)
       
    
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        src_samples  = generators_samples[0]
        dst_samples  = generators_samples[1]

        feed = [src_samples[0], dst_samples[0] ] + \
               src_samples[1:1+self.ms_count*2] + \
               dst_samples[1:1+self.ms_count*2]
               
        src_loss, dst_loss, = self.src_dst_train (feed)
            
        if self.options['learn_mask']:
            feed = [ src_samples[0], dst_samples[0] ] + \
                   src_samples[1+self.ms_count:1+self.ms_count*2] + \
                   dst_samples[1+self.ms_count:1+self.ms_count*2]
            src_mask_loss, dst_mask_loss, = self.src_dst_mask_train (feed)
        
        return ( ('src_loss', src_loss), ('dst_loss', dst_loss) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]

        if self.options['learn_mask']:
            S, D, SS, DD, SD, SDM = [ np.clip(x, 0.0, 1.0) for x in ([test_A,test_B] + self.AE_view ([test_A, test_B]) ) ]
            SDM, = [ np.repeat (x, (3,), -1) for x in [SDM] ]
        else:
            S, D, SS, DD, SD, = [ np.clip(x, 0.0, 1.0) for x in ([test_A,test_B] + self.AE_view ([test_A, test_B]) ) ]

        st = []
        for i in range(0, len(test_A)):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            #if self.options['learn_mask']:
            #    ar += (SDM[i],)            
            st.append ( np.concatenate ( ar, axis=1) )
        
        return [ ('SAE', np.concatenate (st, axis=0 )), ]
    
    def predictor_func (self, face):
    
        prd = [ x[0] for x in self.AE_convert ( [ face[np.newaxis,:,:,0:3] ] ) ]

        if not self.options['learn_mask']:
            prd += [ face[...,3:4] ] 
        
        return np.concatenate ( prd, -1 )
        
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30 if self.options['face_type'] == 'f' else 100
        base_blur_mask_modifier = 0 if self.options['face_type'] == 'f' else 100
        
        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 100 if (self.options['face_style_power'] or self.options['bg_style_power']) and \
                                                self.options['face_type'] == 'f' else 0
        
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF
        
        from converters import ConverterMasked
        return ConverterMasked(self.predictor_func, 
                               predictor_input_size=self.options['resolution'],
                               output_size=self.options['resolution'],
                               face_type=face_type,
                               default_mode = 1 if self.options['face_style_power'] or self.options['bg_style_power'] else 4,
                               base_erode_mask_modifier=base_erode_mask_modifier,
                               base_blur_mask_modifier=base_blur_mask_modifier,
                               default_erode_mask_modifier=default_erode_mask_modifier,
                               default_blur_mask_modifier=default_blur_mask_modifier,
                               clip_hborder_mask_per=0.0625 if self.options['face_type'] == 'f' else 0)
                               
    @staticmethod
    def initialize_nn_functions():
        exec (nnlib.import_all(), locals(), globals())
        
        class ResidualBlock(object):
            def __init__(self, filters, kernel_size=3, padding='same', use_reflection_padding=False):
                self.filters = filters
                self.kernel_size = kernel_size
                self.padding = padding #if not use_reflection_padding else 'valid'
                self.use_reflection_padding = use_reflection_padding
                
            def __call__(self, inp):
                var_x = LeakyReLU(alpha=0.2)(inp)
                
                #if self.use_reflection_padding:
                #    #var_x = ReflectionPadding2D(stride=1, kernel_size=kernel_size)(var_x)

                var_x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding, kernel_initializer=RandomNormal(0, 0.02) )(var_x)                
                var_x = LeakyReLU(alpha=0.2)(var_x)
                
                #if self.use_reflection_padding:
                #    #var_x = ReflectionPadding2D(stride=1, kernel_size=kernel_size)(var_x)
                    
                var_x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding, kernel_initializer=RandomNormal(0, 0.02) )(var_x)
                var_x = Scale(gamma_init=keras.initializers.Constant(value=0.1))(var_x)
                var_x = Add()([var_x, inp])
                var_x = LeakyReLU(alpha=0.2)(var_x)
                return var_x
        SAEModel.ResidualBlock = ResidualBlock

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, kernel_size=5, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x))
            return func 
        SAEModel.downscale = downscale
        
        def downscale_sep (dim):
            def func(x):
                return LeakyReLU(0.1)(SeparableConv2D(dim, kernel_size=5, strides=2, padding='same', depthwise_initializer=RandomNormal(0, 0.02), pointwise_initializer=RandomNormal(0, 0.02) )(x))
            return func 
        SAEModel.downscale_sep = downscale_sep
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(0, 0.02) )(x)))
            return func 
        SAEModel.upscale = upscale
        
        def to_bgr (output_nc):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding='same', activation='sigmoid', kernel_initializer=RandomNormal(0, 0.02) )(x)
            return func
        SAEModel.to_bgr = to_bgr
        
            
    @staticmethod
    def LIAEEncFlow(resolution, light_enc, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        upscale = SAEModel.upscale
        downscale = SAEModel.downscale
        downscale_sep = SAEModel.downscale_sep
    
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
        upscale = SAEModel.upscale
        lowest_dense_res=resolution // 16
        
        def func(input):   
            x = input[0]
            x = Dense(ae_dims)(x)
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims*2)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims*2))(x)
            x = upscale(ae_dims*2)(x)
            return x
        return func
        
    @staticmethod
    def LIAEDecFlow(output_nc,ed_ch_dims=21, multiscale_count=1):
        exec (nnlib.import_all(), locals(), globals())
        upscale = SAEModel.upscale
        to_bgr = SAEModel.to_bgr
        ed_dims = output_nc * ed_ch_dims
            
        def func(input):   
            x = input[0]
            
            outputs = []
            x1     = upscale(ed_dims*8)( x )       
            
            if multiscale_count >= 3:
                outputs += [ to_bgr(output_nc) ( x1 ) ]  
                
            x2     = upscale(ed_dims*4)( x1 )    
            
            if multiscale_count >= 2:
                outputs += [ to_bgr(output_nc) ( x2 ) ]
                
            x3     = upscale(ed_dims*2)( x2 )
            
            outputs += [ to_bgr(output_nc) ( x3 ) ]
        
            return outputs
        return func
    
    @staticmethod
    def DFEncFlow(resolution, light_enc, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        upscale = SAEModel.upscale
        downscale = SAEModel.downscale
        downscale_sep = SAEModel.downscale_sep
        lowest_dense_res = resolution // 16

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
    def DFDecFlow(output_nc, ed_ch_dims=21, multiscale_count=1):
        exec (nnlib.import_all(), locals(), globals())
        upscale = SAEModel.upscale
        to_bgr = SAEModel.to_bgr
        ed_dims = output_nc * ed_ch_dims

        def func(input):   
            x = input[0]
            
            outputs = []
            x1     = upscale(ed_dims*8)( x )       
            
            if multiscale_count >= 3:
                outputs += [ to_bgr(output_nc) ( x1 ) ]  
                
            x2     = upscale(ed_dims*4)( x1 )    
            
            if multiscale_count >= 2:
                outputs += [ to_bgr(output_nc) ( x2 ) ]
                
            x3     = upscale(ed_dims*2)( x2 )
            
            outputs += [ to_bgr(output_nc) ( x3 ) ]
        
            return outputs            
        return func
        
    
            
    @staticmethod
    def VGEncFlow(resolution, light_enc, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        upscale = SAEModel.upscale
        downscale = SAEModel.downscale
        downscale_sep = SAEModel.downscale_sep
        ResidualBlock = SAEModel.ResidualBlock
        lowest_dense_res = resolution // 16
            
        def func(input):
            x = input
            ed_dims = K.int_shape(input)[-1]*ed_ch_dims
            while np.modf(ed_dims / 4)[0] != 0.0:
                ed_dims -= 1
            
            in_conv_filters = ed_dims# if resolution <= 128 else ed_dims + (resolution//128)*ed_ch_dims
            
            x = tmp_x = Conv2D (in_conv_filters, kernel_size=5, strides=2, padding='same') (x)

            for _ in range ( 8 if light_enc else 16 ):
                x = ResidualBlock(ed_dims)(x)
                
            x = Add()([x, tmp_x])

            x = downscale(ed_dims)(x)
            x = SubpixelUpscaler()(x)
            
            x = downscale(ed_dims)(x)
            x = SubpixelUpscaler()(x)
            
            x = downscale(ed_dims)(x)           
            if light_enc:
                x = downscale_sep (ed_dims*2)(x)
            else:
                x = downscale (ed_dims*2)(x)
                
            x = downscale(ed_dims*4)(x)
            
            if light_enc:
                x = downscale_sep (ed_dims*8)(x)
            else:
                x = downscale (ed_dims*8)(x)
            
            x = Dense(ae_dims)(Flatten()(x))
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
            x = upscale(ae_dims)(x)
            return x
            
        return func
        
    @staticmethod
    def VGDecFlow(output_nc, ed_ch_dims=21, multiscale_count=1):
        exec (nnlib.import_all(), locals(), globals())
        upscale = SAEModel.upscale        
        to_bgr = SAEModel.to_bgr
        ResidualBlock = SAEModel.ResidualBlock
        ed_dims = output_nc * ed_ch_dims
  
        def func(input):
            x = input[0]
            
            x = upscale( ed_dims*8 )(x)
            x = ResidualBlock( ed_dims*8 )(x)
            
            x = upscale( ed_dims*4 )(x)
            x = ResidualBlock( ed_dims*4 )(x)
            
            x = upscale( ed_dims*2 )(x)
            x = ResidualBlock( ed_dims*2 )(x)
        
            x = to_bgr(output_nc) (x)        
            return x
            
        return func
        
Model = SAEModel


        
# 'worst' sample booster gives no good result, or I dont know how to filter worst samples properly.
#
##gathering array of sample_losses
#self.src_sample_losses += [[src_sample_idxs[i], src_sample_losses[i]] for i in range(self.batch_size) ]
#self.dst_sample_losses += [[dst_sample_idxs[i], dst_sample_losses[i]] for i in range(self.batch_size) ]
#
#if len(self.src_sample_losses) >= 128: #array is big enough
#    #fetching idxs which losses are bigger than average
#    x = np.array (self.src_sample_losses)
#    self.src_sample_losses = []
#    b = x[:,1]
#    idxs = (x[:,0][ np.argwhere ( b [ b > (np.mean(b)+np.std(b)) ] )[:,0] ]).astype(np.uint)
#    generators_list[0].repeat_sample_idxs(idxs) #ask generator to repeat these sample idxs
#    print ("src repeated %d" % (len(idxs)) )
#    
#if len(self.dst_sample_losses) >= 128: #array is big enough
#    #fetching idxs which losses are bigger than average
#    x = np.array (self.dst_sample_losses)
#    self.dst_sample_losses = []
#    b = x[:,1]
#    idxs = (x[:,0][ np.argwhere ( b [ b > (np.mean(b)+np.std(b)) ] )[:,0] ]).astype(np.uint)
#    generators_list[1].repeat_sample_idxs(idxs) #ask generator to repeat these sample idxs
#    print ("dst repeated %d" % (len(idxs)) )