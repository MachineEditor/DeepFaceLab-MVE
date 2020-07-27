import multiprocessing
import operator
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

class SAEHDModel(ModelBase):

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()

        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb

        if lowest_vram >= 4:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        yn_str = {True:'y',False:'n'}
        min_res = 64
        max_res = 640

        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 128)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'f')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        archi = self.load_or_def_option('archi', 'df')
        archi = {'dfuhd':'df-u','liaeuhd':'liae-u'}.get(archi, archi) #backward comp
        default_archi              = self.options['archi'] = archi

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_masked_training    = self.options['masked_training']    = self.load_or_def_option('masked_training', True)
        default_eyes_prio          = self.options['eyes_prio']          = self.load_or_def_option('eyes_prio', False)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)

        lr_dropout = self.load_or_def_option('lr_dropout', 'n')
        lr_dropout = {True:'y', False:'n'}.get(lr_dropout, lr_dropout) #backward comp
        default_lr_dropout         = self.options['lr_dropout'] = lr_dropout

        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_true_face_power    = self.options['true_face_power']    = self.load_or_def_option('true_face_power', 0.0)
        default_face_style_power   = self.options['face_style_power']   = self.load_or_def_option('face_style_power', 0.0)
        default_bg_style_power     = self.options['bg_style_power']     = self.load_or_def_option('bg_style_power', 0.0)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_flip()
            self.ask_batch_size(suggest_batch_size)

        if self.is_first_run():
            resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi.")
            resolution = np.clip ( (resolution // 16) * 16, min_res, max_res)
            self.options['resolution'] = resolution
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()

            while True:
                archi = io.input_str ("AE architecture", default_archi, help_message=\
"""
'df' keeps more identity-preserved face.
'liae' can fix overly different face shapes.
'-u' increased likeness of the face.
'-d' (experimental) doubling the resolution using the same computation cost.
Examples: df, liae, df-d, df-ud, liae-ud, ...
""").lower()

                archi_split = archi.split('-')

                if len(archi_split) == 2:
                    archi_type, archi_opts = archi_split
                elif len(archi_split) == 1:
                    archi_type, archi_opts = archi_split[0], None
                else:
                    continue

                if archi_type not in ['df', 'liae']:
                    continue

                if archi_opts is not None:
                    if len(archi_opts) == 0:
                        continue
                    if len([ 1 for opt in archi_opts if opt not in ['u','d'] ]) != 0:
                        continue

                    if 'd' in archi_opts:
                        self.options['resolution'] = np.clip ( (self.options['resolution'] // 32) * 32, min_res, max_res)

                break
            self.options['archi'] = archi

        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if self.options['face_type'] == 'wf' or self.options['face_type'] == 'head':
                self.options['masked_training']  = io.input_bool ("Masked training", default_masked_training, help_message="This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly.")

            self.options['eyes_prio'] = io.input_bool ("Eyes priority", default_eyes_prio, help_message='Helps to fix eye problems during training like "alien eyes" and wrong eyes direction ( especially on HD architectures ) by forcing the neural network to train eyes with higher priority. before/after https://i.imgur.com/YQHOuSR.jpg ')
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')

        if self.is_first_run() or ask_override:
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 10.0", help_message="Train the network in Generative Adversarial manner. Forces the neural network to learn small details of the face. Enable it only when the face is trained enough and don't disable. Typical value is 0.1"), 0.0, 10.0 )

            if 'df' in self.options['archi']:
                self.options['true_face_power'] = np.clip ( io.input_number ("'True face' power.", default_true_face_power, add_info="0.0000 .. 1.0", help_message="Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01 . Comparison - https://i.imgur.com/czScS9q.png"), 0.0, 1.0 )
            else:
                self.options['true_face_power'] = 0.0

            self.options['face_style_power'] = np.clip ( io.input_number("Face style power", default_face_style_power, add_info="0.0..100.0", help_message="Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse."), 0.0, 100.0 )
            self.options['bg_style_power'] = np.clip ( io.input_number("Background style power", default_bg_style_power, add_info="0.0..100.0", help_message="Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. For whole_face you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0"), 0.0, 100.0 )

            self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. Try all modes to find the best.")
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain, help_message="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly.")

        if self.options['pretrain'] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        self.resolution = resolution = self.options['resolution']
        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]

        eyes_prio = self.options['eyes_prio']

        archi_split = self.options['archi'].split('-')

        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None

        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)

        self.gan_power = gan_power = 0.0 if self.pretrain else self.options['gan_power']
        random_warp = False if self.pretrain else self.options['random_warp']

        if self.pretrain:
            self.options_show_override['gan_power'] = 0.0
            self.options_show_override['random_warp'] = False
            self.options_show_override['lr_dropout'] = 'n'
            self.options_show_override['face_style_power'] = 0.0
            self.options_show_override['bg_style_power'] = 0.0
            self.options_show_override['uniform_yaw'] = True

        masked_training = self.options['masked_training']
        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_ch=3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)
        self.model_filename_list = []

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_srcm_all = tf.placeholder (nn.floatx, mask_shape)
            self.target_dstm_all = tf.placeholder (nn.floatx, mask_shape)

        # Initializing model classes
        model_archi = nn.DeepFakeArchi(resolution, opts=archi_opts)

        with tf.device (models_opt_device):
            if 'df' in archi_type:
                self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
                encoder_out_ch = self.encoder.compute_output_channels ( (nn.floatx, bgr_shape))

                self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
                inter_out_ch = self.inter.compute_output_channels ( (nn.floatx, (None,encoder_out_ch)))

                self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
                self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

                self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                              [self.inter,       'inter.npy'      ],
                                              [self.decoder_src, 'decoder_src.npy'],
                                              [self.decoder_dst, 'decoder_dst.npy']  ]

                if self.is_training:
                    if self.options['true_face_power'] != 0:
                        self.code_discriminator = nn.CodeDiscriminator(ae_dims, code_res=model_archi.Inter.get_code_res()*2, name='dis' )
                        self.model_filename_list += [ [self.code_discriminator, 'code_discriminator.npy'] ]

            elif 'liae' in archi_type:
                self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
                encoder_out_ch = self.encoder.compute_output_channels ( (nn.floatx, bgr_shape))

                self.inter_AB = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_AB')
                self.inter_B  = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_B')

                inter_AB_out_ch = self.inter_AB.compute_output_channels ( (nn.floatx, (None,encoder_out_ch)))
                inter_B_out_ch = self.inter_B.compute_output_channels ( (nn.floatx, (None,encoder_out_ch)))
                inters_out_ch = inter_AB_out_ch+inter_B_out_ch
                self.decoder = model_archi.Decoder(in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder')

                self.model_filename_list += [ [self.encoder,  'encoder.npy'],
                                              [self.inter_AB, 'inter_AB.npy'],
                                              [self.inter_B , 'inter_B.npy'],
                                              [self.decoder , 'decoder.npy'] ]

            if self.is_training:
                if gan_power != 0:
                    self.D_src = nn.UNetPatchDiscriminator(patch_size=resolution//16, in_ch=input_ch, name="D_src")
                    self.model_filename_list += [ [self.D_src, 'D_src_v2.npy'] ]

                # Initialize optimizers
                lr=5e-5
                lr_dropout = 0.3 if self.options['lr_dropout'] in ['y','cpu'] and not self.pretrain else 1.0
                clipnorm = 1.0 if self.options['clipgrad'] else 0.0

                if 'df' in archi_type:
                    self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                elif 'liae' in archi_type:
                    self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter_AB.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()

                self.src_dst_opt = nn.RMSprop(lr=lr, lr_dropout=lr_dropout, clipnorm=clipnorm, name='src_dst_opt')
                self.src_dst_opt.initialize_variables (self.src_dst_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

                if self.options['true_face_power'] != 0:
                    self.D_code_opt = nn.RMSprop(lr=lr, lr_dropout=lr_dropout, clipnorm=clipnorm, name='D_code_opt')
                    self.D_code_opt.initialize_variables ( self.code_discriminator.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                    self.model_filename_list += [ (self.D_code_opt, 'D_code_opt.npy') ]

                if gan_power != 0:
                    self.D_src_dst_opt = nn.RMSprop(lr=lr, lr_dropout=lr_dropout, clipnorm=clipnorm, name='D_src_dst_opt')
                    self.D_src_dst_opt.initialize_variables ( self.D_src.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')#+self.D_src_x2.get_weights()
                    self.model_filename_list += [ (self.D_src_dst_opt, 'D_src_v2_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)


            # Compute losses per GPU
            gpu_pred_src_src_list = []
            gpu_pred_dst_dst_list = []
            gpu_pred_src_dst_list = []
            gpu_pred_src_srcm_list = []
            gpu_pred_dst_dstm_list = []
            gpu_pred_src_dstm_list = []

            gpu_src_losses = []
            gpu_dst_losses = []
            gpu_G_loss_gvs = []
            gpu_D_code_loss_gvs = []
            gpu_D_src_dst_loss_gvs = []
            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):

                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst      = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src      = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst      = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm_all = self.target_srcm_all[batch_slice,:,:,:]
                        gpu_target_dstm_all = self.target_dstm_all[batch_slice,:,:,:]

                    # process model tensors
                    if 'df' in archi_type:
                        gpu_src_code     = self.inter(self.encoder(gpu_warped_src))
                        gpu_dst_code     = self.inter(self.encoder(gpu_warped_dst))
                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)

                    elif 'liae' in archi_type:
                        gpu_src_code = self.encoder (gpu_warped_src)
                        gpu_src_inter_AB_code = self.inter_AB (gpu_src_code)
                        gpu_src_code = tf.concat([gpu_src_inter_AB_code,gpu_src_inter_AB_code], nn.conv2d_ch_axis  )
                        gpu_dst_code = self.encoder (gpu_warped_dst)
                        gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                        gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                        gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )
                        gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis )

                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder(gpu_src_code)
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    # unpack masks from one combined mask
                    gpu_target_srcm      = tf.clip_by_value (gpu_target_srcm_all, 0, 1)
                    gpu_target_dstm      = tf.clip_by_value (gpu_target_dstm_all, 0, 1)
                    gpu_target_srcm_eyes = tf.clip_by_value (gpu_target_srcm_all-1, 0, 1)
                    gpu_target_dstm_eyes = tf.clip_by_value (gpu_target_dstm_all-1, 0, 1)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2

                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )
                    gpu_target_dstm_style_blur = gpu_target_dstm_blur #default style mask is 0.5 on boundary
                    gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_style_masked      = gpu_target_dst*gpu_target_dstm_style_blur
                    gpu_target_dst_style_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_style_blur)

                    gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt  = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_style_masked = gpu_pred_src_dst*gpu_target_dstm_style_blur
                    gpu_psd_target_dst_style_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_style_blur)

                    if resolution < 256:
                        gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    else:
                        gpu_src_loss =  tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])

                    if eyes_prio:
                        gpu_src_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_src*gpu_target_srcm_eyes - gpu_pred_src_src*gpu_target_srcm_eyes ), axis=[1,2,3])

                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    face_style_power = self.options['face_style_power'] / 100.0
                    if face_style_power != 0 and not self.pretrain:
                        gpu_src_loss += nn.style_loss(gpu_psd_target_dst_style_masked, gpu_target_dst_style_masked, gaussian_blur_radius=resolution//16, loss_weight=10000*face_style_power)

                    bg_style_power = self.options['bg_style_power'] / 100.0
                    if bg_style_power != 0 and not self.pretrain:
                        gpu_src_loss += tf.reduce_mean( (10*bg_style_power)*nn.dssim( gpu_psd_target_dst_style_anti_masked,  gpu_target_dst_style_anti_masked, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        gpu_src_loss += tf.reduce_mean( (10*bg_style_power)*tf.square(gpu_psd_target_dst_style_anti_masked - gpu_target_dst_style_anti_masked), axis=[1,2,3] )

                    if resolution < 256:
                        gpu_dst_loss = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    else:
                        gpu_dst_loss = tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                        gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])


                    if eyes_prio:
                        gpu_dst_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_dst*gpu_target_dstm_eyes - gpu_pred_dst_dst*gpu_target_dstm_eyes ), axis=[1,2,3])

                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss

                    def DLoss(labels,logits):
                        return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1,2,3])

                    if self.options['true_face_power'] != 0:
                        gpu_src_code_d = self.code_discriminator( gpu_src_code )
                        gpu_src_code_d_ones  = tf.ones_like (gpu_src_code_d)
                        gpu_src_code_d_zeros = tf.zeros_like(gpu_src_code_d)
                        gpu_dst_code_d = self.code_discriminator( gpu_dst_code )
                        gpu_dst_code_d_ones = tf.ones_like(gpu_dst_code_d)

                        gpu_G_loss += self.options['true_face_power']*DLoss(gpu_src_code_d_ones, gpu_src_code_d)

                        gpu_D_code_loss = (DLoss(gpu_src_code_d_ones , gpu_dst_code_d) + \
                                           DLoss(gpu_src_code_d_zeros, gpu_src_code_d) ) * 0.5

                        gpu_D_code_loss_gvs += [ nn.gradients (gpu_D_code_loss, self.code_discriminator.get_weights() ) ]

                    if gan_power != 0:
                        gpu_pred_src_src_d, \
                        gpu_pred_src_src_d2           = self.D_src(gpu_pred_src_src_masked_opt)

                        gpu_pred_src_src_d_ones  = tf.ones_like (gpu_pred_src_src_d)
                        gpu_pred_src_src_d_zeros = tf.zeros_like(gpu_pred_src_src_d)

                        gpu_pred_src_src_d2_ones  = tf.ones_like (gpu_pred_src_src_d2)
                        gpu_pred_src_src_d2_zeros = tf.zeros_like(gpu_pred_src_src_d2)

                        gpu_target_src_d, \
                        gpu_target_src_d2            = self.D_src(gpu_target_src_masked_opt)

                        gpu_target_src_d_ones    = tf.ones_like(gpu_target_src_d)
                        gpu_target_src_d2_ones    = tf.ones_like(gpu_target_src_d2)

                        gpu_D_src_dst_loss = (DLoss(gpu_target_src_d_ones      , gpu_target_src_d) + \
                                              DLoss(gpu_pred_src_src_d_zeros   , gpu_pred_src_src_d) ) * 0.5 + \
                                             (DLoss(gpu_target_src_d2_ones      , gpu_target_src_d2) + \
                                              DLoss(gpu_pred_src_src_d2_zeros   , gpu_pred_src_src_d2) ) * 0.5

                        gpu_D_src_dst_loss_gvs += [ nn.gradients (gpu_D_src_dst_loss, self.D_src.get_weights() ) ]#+self.D_src_x2.get_weights()

                        gpu_G_loss += gan_power*(DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d)  + \
                                                 DLoss(gpu_pred_src_src_d2_ones, gpu_pred_src_src_d2))

                    gpu_G_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]


            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

                src_loss = tf.concat(gpu_src_losses, 0)
                dst_loss = tf.concat(gpu_dst_losses, 0)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gvs))

                if self.options['true_face_power'] != 0:
                    D_loss_gv_op = self.D_code_opt.get_update_op (nn.average_gv_list(gpu_D_code_loss_gvs))

                if gan_power != 0:
                    src_D_src_dst_loss_gv_op = self.D_src_dst_opt.get_update_op (nn.average_gv_list(gpu_D_src_dst_loss_gvs) )


            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm_all, \
                              warped_dst, target_dst, target_dstm_all):
                s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm_all:target_srcm_all,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm_all:target_dstm_all,
                                                       })
                return s, d
            self.src_dst_train = src_dst_train

            if self.options['true_face_power'] != 0:
                def D_train(warped_src, warped_dst):
                    nn.tf_sess.run ([D_loss_gv_op], feed_dict={self.warped_src: warped_src, self.warped_dst: warped_dst})
                self.D_train = D_train

            if gan_power != 0:
                def D_src_dst_train(warped_src, target_src, target_srcm_all, \
                                    warped_dst, target_dst, target_dstm_all):
                    nn.tf_sess.run ([src_D_src_dst_loss_gv_op], feed_dict={self.warped_src :warped_src,
                                                                           self.target_src :target_src,
                                                                           self.target_srcm_all:target_srcm_all,
                                                                           self.warped_dst :warped_dst,
                                                                           self.target_dst :target_dst,
                                                                           self.target_dstm_all:target_dstm_all})
                self.D_src_dst_train = D_src_dst_train


            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src:warped_src,
                                                    self.warped_dst:warped_dst})
            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
                if 'df' in archi_type:
                    gpu_dst_code     = self.inter(self.encoder(self.warped_dst))
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

                elif 'liae' in archi_type:
                    gpu_dst_code = self.encoder (self.warped_dst)
                    gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                    gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                    gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)
                    gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)

                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)


            def AE_merge( warped_dst):
                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if 'df' in archi_type:
                    if model == self.inter:
                        do_init = True
                elif 'liae' in archi_type:
                    if model == self.inter_AB or model == self.inter_B:
                        do_init = True
            else:
                do_init = self.is_first_run()

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None and not self.pretrain else None

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE_EYES, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE_EYES, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=dst_generators_count )
                             ])

            self.last_src_samples_loss = []
            self.last_dst_samples_loss = []

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % ( 10*(max(1,self.resolution // 64)) ) == 0) or \
               (io.is_colab() and self.iter % 100 == 0)

    #override
    def onTrainOneIter(self):
        if self.get_iter() == 0 and not self.pretrain and not self.pretrain_just_disabled:
            io.log_info('You are training the model from scratch. It is strongly recommended to use a pretrained model to speed up the training and improve the quality.\n')

        bs = self.get_batch_size()

        ( (warped_src, target_src, target_srcm_all), \
          (warped_dst, target_dst, target_dstm_all) ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm_all, warped_dst, target_dst, target_dstm_all)

        for i in range(bs):
            self.last_src_samples_loss.append (  (target_src[i], target_srcm_all[i], src_loss[i] )  )
            self.last_dst_samples_loss.append (  (target_dst[i], target_dstm_all[i], dst_loss[i] )  )

        if len(self.last_src_samples_loss) >= bs*16:
            src_samples_loss = sorted(self.last_src_samples_loss, key=operator.itemgetter(2), reverse=True)
            dst_samples_loss = sorted(self.last_dst_samples_loss, key=operator.itemgetter(2), reverse=True)

            target_src      = np.stack( [ x[0] for x in src_samples_loss[:bs] ] )
            target_srcm_all = np.stack( [ x[1] for x in src_samples_loss[:bs] ] )

            target_dst      = np.stack( [ x[0] for x in dst_samples_loss[:bs] ] )
            target_dstm_all = np.stack( [ x[1] for x in dst_samples_loss[:bs] ] )

            src_loss, dst_loss = self.src_dst_train (target_src, target_src, target_srcm_all, target_dst, target_dst, target_dstm_all)
            self.last_src_samples_loss = []
            self.last_dst_samples_loss = []

        if self.options['true_face_power'] != 0 and not self.pretrain:
            self.D_train (warped_src, warped_dst)

        if self.gan_power != 0:
            self.D_src_dst_train (warped_src, target_src, target_srcm_all, warped_dst, target_dst, target_dstm_all)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )

    #override
    def onGetPreview(self, samples):
        ( (warped_src, target_src, target_srcm_all,),
          (warped_dst, target_dst, target_dstm_all,) ) = samples

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        target_srcm_all, target_dstm_all = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm_all, target_dstm_all] )]

        target_srcm = np.clip(target_srcm_all, 0, 1)
        target_dstm = np.clip(target_dstm_all, 0, 1)

        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        if self.resolution <= 256:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]

                ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('SAEHD masked', np.concatenate (st_m, axis=0 )), ]
        else:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD src-src', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], DD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD dst-dst', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD pred', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                ar = S[i]*target_srcm[i], SS[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked src-src', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                ar = D[i]*target_dstm[i], DD[i]*DDM[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked dst-dst', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                ar = D[i]*target_dstm[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked pred', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = SAEHDModel
