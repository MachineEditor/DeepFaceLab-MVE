import multiprocessing
import operator

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

from pathlib import Path

from utils.label_face import label_face_filename

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

        min_res = 64
        max_res = 640

        default_usefp16            = self.options['use_fp16']           = self.load_or_def_option('use_fp16', False)
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 128)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'f')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_archi              = self.options['archi']              = self.load_or_def_option('archi', 'liae-ud')

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_masked_training    = self.options['masked_training']    = self.load_or_def_option('masked_training', True)
        default_eyes_prio          = self.options['eyes_prio']          = self.load_or_def_option('eyes_prio', False)
        default_mouth_prio         = self.options['mouth_prio']         = self.load_or_def_option('mouth_prio', False)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)

        default_adabelief          = self.options['adabelief']          = self.load_or_def_option('adabelief', True)

        lr_dropout = self.load_or_def_option('lr_dropout', 'n')
        lr_dropout = {True:'y', False:'n'}.get(lr_dropout, lr_dropout) #backward comp
        default_lr_dropout         = self.options['lr_dropout'] = lr_dropout

        default_loss_function      = self.options['loss_function']      = self.load_or_def_option('loss_function', 'SSIM')

        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_random_hsv_power   = self.options['random_hsv_power']   = self.load_or_def_option('random_hsv_power', 0.0)
        default_random_downsample  = self.options['random_downsample']  = self.load_or_def_option('random_downsample', False)
        default_random_noise       = self.options['random_noise']       = self.load_or_def_option('random_noise', False)
        default_random_blur        = self.options['random_blur']        = self.load_or_def_option('random_blur', False)
        default_random_jpeg        = self.options['random_jpeg']        = self.load_or_def_option('random_jpeg', False)
        default_random_shadow      = self.options['random_shadow']      = self.load_or_def_option('random_shadow', 'none')

        default_background_power   = self.options['background_power']   = self.load_or_def_option('background_power', 0.0)
        default_true_face_power    = self.options['true_face_power']    = self.load_or_def_option('true_face_power', 0.0)
        default_face_style_power   = self.options['face_style_power']   = self.load_or_def_option('face_style_power', 0.0)
        default_bg_style_power     = self.options['bg_style_power']     = self.load_or_def_option('bg_style_power', 0.0)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_random_color       = self.options['random_color']       = self.load_or_def_option('random_color', False)
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)
        default_cpu_cap            = self.options['cpu_cap']            = self.load_or_def_option('cpu_cap', 8)
        default_preview_samples    = self.options['preview_samples']    = self.load_or_def_option('preview_samples', 4)
        default_full_preview       = self.options['force_full_preview'] = self.load_or_def_option('force_full_preview', False)
        default_lr                 = self.options['lr']                 = self.load_or_def_option('lr', 5e-5)

        ask_override = False if self.read_from_conf else self.ask_override()
        if self.is_first_run() or ask_override:
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                self.ask_session_name()
                self.ask_autobackup_hour()
                self.ask_maximum_n_backups()
                self.ask_write_preview_history()
                self.options['preview_samples'] = np.clip ( io.input_int ("Number of samples to preview", default_preview_samples, add_info="1 - 16", help_message="Typical fine value is 4"), 1, 16 )
                self.options['force_full_preview'] = io.input_bool ("Use old preview panel", default_full_preview)
                
                self.ask_target_iter()
                self.ask_retraining_samples()
                self.ask_random_src_flip()
                self.ask_random_dst_flip()
                self.ask_batch_size(suggest_batch_size)
                self.options['use_fp16'] = io.input_bool ("Use fp16", default_usefp16, help_message='Increases training/inference speed, reduces model size. Model may crash. Enable it after 1-5k iters.')
                self.options['cpu_cap'] = np.clip ( io.input_int ("Max cpu cores to use.", default_cpu_cap, add_info="1 - 256", help_message="Typical fine value is 8"), 1, 256 )


        if self.is_first_run():
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi.")
                resolution = np.clip ( (resolution // 16) * 16, min_res, max_res)
                self.options['resolution'] = resolution
                self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head', 'custom'], help_message="Half / mid face / full face / whole face / head / custom. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()

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
                        if len([ 1 for opt in archi_opts if opt not in ['u','d','t','c'] ]) != 0:
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
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )

                e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
                self.options['e_dims'] = e_dims + e_dims % 2

                d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
                self.options['d_dims'] = d_dims + d_dims % 2

                d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
                self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                if self.options['face_type'] == 'wf' or self.options['face_type'] == 'head' or self.options['face_type'] == 'custom':
                    self.options['masked_training']  = io.input_bool ("Masked training", default_masked_training, help_message="This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly.")

                self.options['eyes_prio'] = io.input_bool ("Eyes priority", default_eyes_prio, help_message='Helps to fix eye problems during training like "alien eyes" and wrong eyes direction ( especially on HD architectures ) by forcing the neural network to train eyes with higher priority. before/after https://i.imgur.com/YQHOuSR.jpg ')
                self.options['mouth_prio'] = io.input_bool ("Mouth priority", default_mouth_prio, help_message='Helps to fix mouth problems during training by forcing the neural network to train mouth with higher priority similar to eyes ')

                self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')
                self.options['blur_out_mask'] = io.input_bool ("Blur out mask", default_blur_out_mask, help_message='Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required.')

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)
        default_gan_smoothing      = self.options['gan_smoothing']      = self.load_or_def_option('gan_smoothing', 0.1)
        default_gan_noise          = self.options['gan_noise']          = self.load_or_def_option('gan_noise', 0.0)

        if self.is_first_run() or ask_override:
            if (self.read_from_conf and not self.config_file_exists) or not self.read_from_conf:
                self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

                self.options['adabelief'] = io.input_bool ("Use AdaBelief optimizer?", default_adabelief, help_message="Use AdaBelief optimizer. It requires more VRAM, but the accuracy and the generalization of the model is higher.")

                self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

                self.options['loss_function'] = io.input_str(f"Loss function", default_loss_function, ['SSIM', 'MS-SSIM', 'MS-SSIM+L1'],
                                                            help_message="Change loss function used for image quality assessment.")
                
                self.options['lr'] = np.clip (io.input_number("Learning rate", default_lr, add_info="0.0 .. 1.0", help_message="Learning rate: typical fine value 5e-5"), 0.0, 1)

                self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

                self.options['random_hsv_power'] = np.clip ( io.input_number ("Random hue/saturation/light intensity", default_random_hsv_power, add_info="0.0 .. 0.3", help_message="Random hue/saturation/light intensity applied to the src face set only at the input of the neural network. Stabilizes color perturbations during face swapping. Reduces the quality of the color transfer by selecting the closest one in the src faceset. Thus the src faceset must be diverse enough. Typical fine value is 0.05"), 0.0, 0.3 )

                self.options['random_downsample'] = io.input_bool("Enable random downsample of samples", default_random_downsample, help_message="")
                self.options['random_noise'] = io.input_bool("Enable random noise added to samples", default_random_noise, help_message="")
                self.options['random_blur'] = io.input_bool("Enable random blur of samples", default_random_blur, help_message="")
                self.options['random_jpeg'] = io.input_bool("Enable random jpeg compression of samples", default_random_jpeg, help_message="")
                self.options['random_shadow'] = io.input_str('Enable random shadows and highlights of samples', default_random_shadow, ['none','src','dst','all'], help_message="Helps to create shadows in dataset. Use src if you src dataset has lack of shadows/different lighting situations; dst to help generalization; all for both reason.")
        
                self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 10.0", help_message="Train the network in Generative Adversarial manner. Forces the neural network to learn small details of the face. Enable it only when the face is trained enough and don't disable. Typical value is 0.1"), 0.0, 10.0 )

                if self.options['gan_power'] != 0.0:

                    gan_patch_size = np.clip ( io.input_int("GAN patch size", default_gan_patch_size, add_info="3-640", help_message="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8." ), 3, 640 )
                    self.options['gan_patch_size'] = gan_patch_size

                    gan_dims = np.clip ( io.input_int("GAN dimensions", default_gan_dims, add_info="4-64", help_message="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16." ), 4, 64 )
                    self.options['gan_dims'] = gan_dims

                    self.options['gan_smoothing'] = np.clip ( io.input_number("GAN label smoothing", default_gan_smoothing, add_info="0 - 0.5", help_message="Uses soft labels with values slightly off from 0/1 for GAN, has a regularizing effect"), 0, 0.5)
                    self.options['gan_noise'] = np.clip ( io.input_number("GAN noisy labels", default_gan_noise, add_info="0 - 0.5", help_message="Marks some images with the wrong label, helps prevent collapse"), 0, 0.5)

                if 'df' in self.options['archi']:
                    self.options['true_face_power'] = np.clip ( io.input_number ("'True face' power.", default_true_face_power, add_info="0.0000 .. 1.0", help_message="Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01 . Comparison - https://i.imgur.com/czScS9q.png"), 0.0, 1.0 )
                else:
                    self.options['true_face_power'] = 0.0

                self.options['background_power'] = np.clip ( io.input_number("Background power", default_background_power, add_info="0.0..1.0", help_message="Learn the area outside of the mask. Helps smooth out area near the mask boundaries. Can be used at any time"), 0.0, 1.0 )

                self.options['face_style_power'] = np.clip ( io.input_number("Face style power", default_face_style_power, add_info="0.0..100.0", help_message="Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse."), 0.0, 100.0 )
                self.options['bg_style_power'] = np.clip ( io.input_number("Background style power", default_bg_style_power, add_info="0.0..100.0", help_message="Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. For whole_face you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0"), 0.0, 100.0 )

                self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot', 'fs-aug'], help_message="Change color distribution of src samples close to dst samples. Try all modes to find the best. FS aug adds random color to dst and src")
                self.options['random_color'] = io.input_bool ("Random color", default_random_color, help_message="Samples are randomly rotated around the L axis in LAB colorspace, helps generalize training")
                self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

                self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain, help_message="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly. Forces random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y")

        if self.options['pretrain'] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

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
                          'custom' : FaceType.CUSTOM,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]

        eyes_prio = self.options['eyes_prio']
        mouth_prio = self.options['mouth_prio']

        archi_split = self.options['archi'].split('-')

        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None

        self.archi_type = archi_type

        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)

        adabelief = self.options['adabelief']
        
        use_fp16 = self.options['use_fp16']
        if self.is_exporting:
            use_fp16 = io.input_bool ("Export quantized?", False, help_message='Makes the exported model faster. If you have problems, disable this option.')

        self.gan_power = gan_power = 0.0 if self.pretrain else self.options['gan_power']
        random_warp = False if self.pretrain else self.options['random_warp']
        random_src_flip = self.random_src_flip if not self.pretrain else True
        random_dst_flip = self.random_dst_flip if not self.pretrain else True
        random_hsv_power = self.options['random_hsv_power'] if not self.pretrain else 0.0
        blur_out_mask = self.options['blur_out_mask']
        
        if self.pretrain:
            self.options_show_override['lr_dropout'] = 'n'
            self.options_show_override['random_warp'] = False
            self.options_show_override['gan_power'] = 0.0
            self.options_show_override['random_hsv_power'] = 0.0
            self.options_show_override['face_style_power'] = 0.0
            self.options_show_override['bg_style_power'] = 0.0
            self.options_show_override['uniform_yaw'] = True

        masked_training = self.options['masked_training']
        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        random_shadow_src = True if self.options['random_shadow'] in ['all', 'src'] else False
        random_shadow_dst = True if self.options['random_shadow'] in ['all', 'dst'] else False

        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = nn.tf_default_device_name if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_ch=3
        bgr_shape = self.bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)
        self.model_filename_list = []

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape, name='warped_src')
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape, name='warped_dst')

            self.target_src = tf.placeholder (nn.floatx, bgr_shape, name='target_src')
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape, name='target_dst')

            self.target_srcm    = tf.placeholder (nn.floatx, mask_shape, name='target_srcm')
            self.target_srcm_em = tf.placeholder (nn.floatx, mask_shape, name='target_srcm_em')
            self.target_dstm    = tf.placeholder (nn.floatx, mask_shape, name='target_dstm')
            self.target_dstm_em = tf.placeholder (nn.floatx, mask_shape, name='target_dstm_em')

        # Initializing model classes
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=use_fp16, opts=archi_opts)

        with tf.device (models_opt_device):
            if 'df' in archi_type:
                self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
                encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

                self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
                inter_out_ch = self.inter.get_out_ch()

                self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
                self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

                self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                              [self.inter,       'inter.npy'      ],
                                              [self.decoder_src, 'decoder_src.npy'],
                                              [self.decoder_dst, 'decoder_dst.npy']  ]

                if self.is_training:
                    if self.options['true_face_power'] != 0:
                        self.code_discriminator = nn.CodeDiscriminator(ae_dims, code_res=self.inter.get_out_res(), name='dis' )
                        self.model_filename_list += [ [self.code_discriminator, 'code_discriminator.npy'] ]

            elif 'liae' in archi_type:
                self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
                encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

                self.inter_AB = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_AB')
                self.inter_B  = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_B')

                inter_out_ch = self.inter_AB.get_out_ch()
                inters_out_ch = inter_out_ch*2
                self.decoder = model_archi.Decoder(in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder')

                self.model_filename_list += [ [self.encoder,  'encoder.npy'],
                                              [self.inter_AB, 'inter_AB.npy'],
                                              [self.inter_B , 'inter_B.npy'],
                                              [self.decoder , 'decoder.npy'] ]

            if self.is_training:
                if gan_power != 0:
                    self.D_src = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], use_fp16=self.options['use_fp16'], name="D_src")
                    self.model_filename_list += [ [self.D_src, 'GAN.npy'] ]

                # Initialize optimizers
                lr = self.options['lr']
                
                if self.options['lr_dropout'] in ['y','cpu'] and not self.pretrain:
                    lr_cos = 500
                    lr_dropout = 0.3
                else:
                    lr_cos = 0
                    lr_dropout = 1.0
                OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
                clipnorm = 1.0 if self.options['clipgrad'] else 0.0

                if 'df' in archi_type:
                    self.src_dst_saveable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                    self.src_dst_trainable_weights = self.src_dst_saveable_weights
                elif 'liae' in archi_type:
                    self.src_dst_saveable_weights = self.encoder.get_weights() + self.inter_AB.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()
                    if random_warp:
                        self.src_dst_trainable_weights = self.src_dst_saveable_weights
                    else:
                        self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter_B.get_weights() + self.decoder.get_weights()

                self.src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
                self.src_dst_opt.initialize_variables (self.src_dst_saveable_weights, vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

                if self.options['true_face_power'] != 0:
                    self.D_code_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='D_code_opt')
                    self.D_code_opt.initialize_variables ( self.code_discriminator.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                    self.model_filename_list += [ (self.D_code_opt, 'D_code_opt.npy') ]

                if gan_power != 0:
                    self.D_src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                    self.D_src_dst_opt.initialize_variables ( self.D_src.get_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')#+self.D_src_x2.get_weights()
                    self.model_filename_list += [ (self.D_src_dst_opt, 'GAN_opt.npy') ]


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
                with tf.device( f'/{devices[gpu_id].tf_dev_type}:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst      = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src      = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst      = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm_all = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_srcm_em = self.target_srcm_em[batch_slice,:,:,:]
                        gpu_target_dstm_all = self.target_dstm[batch_slice,:,:,:]
                        gpu_target_dstm_em = self.target_dstm_em[batch_slice,:,:,:]

                    gpu_target_srcm_anti = 1-gpu_target_srcm_all
                    gpu_target_dstm_anti = 1-gpu_target_dstm_all

                    if blur_out_mask:
                        sigma = resolution / 128

                        x = nn.gaussian_blur(gpu_target_src*gpu_target_srcm_anti, sigma)
                        y = 1-nn.gaussian_blur(gpu_target_srcm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_src = gpu_target_src*gpu_target_srcm_all + (x/y)*gpu_target_srcm_anti
                        
                        x = nn.gaussian_blur(gpu_target_dst*gpu_target_dstm_anti, sigma)
                        y = 1-nn.gaussian_blur(gpu_target_dstm_all, sigma)
                        y = tf.where(tf.equal(y, 0), tf.ones_like(y), y)
                        gpu_target_dst = gpu_target_dst*gpu_target_dstm_all + (x/y)*gpu_target_dstm_anti


                    # process model tensors
                    if 'df' in archi_type:
                        gpu_src_code     = self.inter(self.encoder(gpu_warped_src))
                        gpu_dst_code     = self.inter(self.encoder(gpu_warped_dst))
                        gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                        gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                        gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                        gpu_pred_src_dst_no_code_grad, _ = self.decoder_src(tf.stop_gradient(gpu_dst_code))

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
                        gpu_pred_src_dst_no_code_grad, _ = self.decoder(tf.stop_gradient(gpu_src_dst_code))

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    # unpack masks from one combined mask
                    gpu_target_srcm      = tf.clip_by_value (gpu_target_srcm_all, 0, 1)
                    gpu_target_dstm      = tf.clip_by_value (gpu_target_dstm_all, 0, 1)
                    gpu_target_srcm_eye_mouth = tf.clip_by_value (gpu_target_srcm_em-1, 0, 1)
                    gpu_target_dstm_eye_mouth = tf.clip_by_value (gpu_target_dstm_em-1, 0, 1)
                    gpu_target_srcm_mouth = tf.clip_by_value (gpu_target_srcm_em-2, 0, 1)
                    gpu_target_dstm_mouth = tf.clip_by_value (gpu_target_dstm_em-2, 0, 1)
                    gpu_target_srcm_eyes = tf.clip_by_value (gpu_target_srcm_eye_mouth-gpu_target_srcm_mouth, 0, 1)
                    gpu_target_dstm_eyes = tf.clip_by_value (gpu_target_dstm_eye_mouth-gpu_target_dstm_mouth, 0, 1)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_srcm_blur = tf.clip_by_value(gpu_target_srcm_blur, 0, 0.5) * 2
                    gpu_target_srcm_anti_blur = 1.0-gpu_target_srcm_blur

                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )
                    gpu_target_dstm_blur = tf.clip_by_value(gpu_target_dstm_blur, 0, 0.5) * 2

                    gpu_style_mask_blur = nn.gaussian_blur(gpu_pred_src_dstm*gpu_pred_dst_dstm,  max(1, resolution // 32) )
                    gpu_style_mask_blur = tf.stop_gradient(tf.clip_by_value(gpu_target_srcm_blur, 0, 1.0))
                    gpu_style_mask_anti_blur = 1.0 - gpu_style_mask_blur

                    gpu_target_dst_masked = gpu_target_dst*gpu_target_dstm_blur

                    gpu_target_src_anti_masked = gpu_target_src*gpu_target_srcm_anti_blur
                    gpu_pred_src_src_anti_masked = gpu_pred_src_src*gpu_target_srcm_anti_blur

                    gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt  = gpu_target_dst_masked if masked_training else gpu_target_dst
                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    if self.options['loss_function'] == 'MS-SSIM':
                        gpu_src_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0)
                        gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    elif self.options['loss_function'] == 'MS-SSIM+L1':
                        gpu_src_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0)
                    else:
                        if resolution < 256:
                            gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        else:
                            gpu_src_loss =  tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                            gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                        gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])

                    if eyes_prio or mouth_prio:
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask = gpu_target_srcm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask = gpu_target_srcm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask = gpu_target_srcm_mouth

                        gpu_src_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_src*gpu_target_part_mask - gpu_pred_src_src*gpu_target_part_mask ), axis=[1,2,3])

                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    if self.options['background_power'] > 0:
                        bg_factor = self.options['background_power']

                        if self.options['loss_function'] == 'MS-SSIM':
                            gpu_src_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_src, gpu_pred_src_src, max_val=1.0)
                            gpu_src_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_src - gpu_pred_src_src ), axis=[1,2,3])
                        elif self.options['loss_function'] == 'MS-SSIM+L1':
                            gpu_src_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_src, gpu_pred_src_src, max_val=1.0)
                        else:
                            if resolution < 256:
                                gpu_src_loss +=  bg_factor * tf.reduce_mean ( 10*nn.dssim(gpu_target_src, gpu_pred_src_src, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                            else:
                                gpu_src_loss +=  bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_src, gpu_pred_src_src, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                                gpu_src_loss += bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_src, gpu_pred_src_src, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                            gpu_src_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_src - gpu_pred_src_src ), axis=[1,2,3])

                    face_style_power = self.options['face_style_power'] / 100.0
                    if face_style_power != 0 and not self.pretrain:
                        gpu_src_loss += nn.style_loss(gpu_pred_src_dst_no_code_grad*tf.stop_gradient(gpu_pred_src_dstm), tf.stop_gradient(gpu_pred_dst_dst*gpu_pred_dst_dstm), gaussian_blur_radius=resolution//8, loss_weight=10000*face_style_power)

                    bg_style_power = self.options['bg_style_power'] / 100.0
                    if bg_style_power != 0 and not self.pretrain:
                        gpu_target_dst_style_anti_masked = gpu_target_dst*gpu_style_mask_anti_blur
                        gpu_psd_style_anti_masked = gpu_pred_src_dst*gpu_style_mask_anti_blur

                        gpu_src_loss += tf.reduce_mean( (10*bg_style_power)*nn.dssim( gpu_psd_style_anti_masked,  gpu_target_dst_style_anti_masked, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        gpu_src_loss += tf.reduce_mean( (10*bg_style_power)*tf.square(gpu_psd_style_anti_masked - gpu_target_dst_style_anti_masked), axis=[1,2,3] )

                    if self.options['loss_function'] == 'MS-SSIM':
                        gpu_dst_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0)
                        gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    elif self.options['loss_function'] == 'MS-SSIM+L1':
                        gpu_dst_loss = 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0)
                    else:
                        if resolution < 256:
                            gpu_dst_loss = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                        else:
                            gpu_dst_loss = tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                            gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/23.2) ), axis=[1])
                        gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])

                    if eyes_prio or mouth_prio:
                        if eyes_prio and mouth_prio:
                            gpu_target_part_mask = gpu_target_dstm_eye_mouth
                        elif eyes_prio:
                            gpu_target_part_mask = gpu_target_dstm_eyes
                        elif mouth_prio:
                            gpu_target_part_mask = gpu_target_dstm_mouth

                        gpu_dst_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_dst*gpu_target_part_mask - gpu_pred_dst_dst*gpu_target_part_mask ), axis=[1,2,3])

                    if self.options['background_power'] > 0:
                        bg_factor = self.options['background_power']

                        if self.options['loss_function'] == 'MS-SSIM':
                            gpu_dst_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution)(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0)
                            gpu_dst_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_dst - gpu_pred_dst_dst ), axis=[1,2,3])
                        elif self.options['loss_function'] == 'MS-SSIM+L1':
                            gpu_dst_loss += bg_factor * 10 * nn.MsSsim(bs_per_gpu, input_ch, resolution, use_l1=True)(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0)
                        else:
                            if resolution < 256:
                                gpu_dst_loss +=  bg_factor * tf.reduce_mean ( 10*nn.dssim(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                            else:
                                gpu_dst_loss +=  bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                                gpu_dst_loss += bg_factor * tf.reduce_mean ( 5*nn.dssim(gpu_target_dst, gpu_pred_dst_dst, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                            gpu_dst_loss += bg_factor * tf.reduce_mean ( 10*tf.square ( gpu_target_dst - gpu_pred_dst_dst ), axis=[1,2,3])

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

                        gpu_D_code_loss = (DLoss(gpu_dst_code_d_ones , gpu_dst_code_d) + \
                                           DLoss(gpu_src_code_d_zeros, gpu_src_code_d) ) * 0.5

                        gpu_D_code_loss_gvs += [ nn.gradients (gpu_D_code_loss, self.code_discriminator.get_weights() ) ]

                    if gan_power != 0:
                        gpu_pred_src_src_d, \
                        gpu_pred_src_src_d2           = self.D_src(gpu_pred_src_src_masked_opt)

                        def get_smooth_noisy_labels(label, tensor, smoothing=0.1, noise=0.05):
                            num_labels = self.batch_size
                            for d in tensor.get_shape().as_list()[1:]:
                                num_labels *= d

                            probs = tf.math.log([[noise, 1-noise]]) if label == 1 else tf.math.log([[1-noise, noise]])
                            x = tf.random.categorical(probs, num_labels)
                            x = tf.cast(x, tf.float32)
                            x = tf.math.scalar_mul(1-smoothing, x)
                            # x = x + (smoothing/num_labels)
                            x = tf.reshape(x, (self.batch_size,) + tuple(tensor.get_shape().as_list()[1:]))
                            return x

                        smoothing = self.options['gan_smoothing']
                        noise = self.options['gan_noise']

                        gpu_pred_src_src_d_ones = tf.ones_like(gpu_pred_src_src_d)
                        gpu_pred_src_src_d2_ones = tf.ones_like(gpu_pred_src_src_d2)

                        gpu_pred_src_src_d_smooth_zeros = get_smooth_noisy_labels(0, gpu_pred_src_src_d, smoothing=smoothing, noise=noise)
                        gpu_pred_src_src_d2_smooth_zeros = get_smooth_noisy_labels(0, gpu_pred_src_src_d2, smoothing=smoothing, noise=noise)

                        gpu_target_src_d, gpu_target_src_d2 = self.D_src(gpu_target_src_masked_opt)

                        gpu_target_src_d_smooth_ones = get_smooth_noisy_labels(1, gpu_target_src_d, smoothing=smoothing, noise=noise)
                        gpu_target_src_d2_smooth_ones = get_smooth_noisy_labels(1, gpu_target_src_d2, smoothing=smoothing, noise=noise)

                        gpu_D_src_dst_loss = DLoss(gpu_target_src_d_smooth_ones, gpu_target_src_d) \
                                             + DLoss(gpu_pred_src_src_d_smooth_zeros, gpu_pred_src_src_d) \
                                             + DLoss(gpu_target_src_d2_smooth_ones, gpu_target_src_d2) \
                                             + DLoss(gpu_pred_src_src_d2_smooth_zeros, gpu_pred_src_src_d2)

                        gpu_D_src_dst_loss_gvs += [ nn.gradients (gpu_D_src_dst_loss, self.D_src.get_weights() ) ]#+self.D_src_x2.get_weights()

                        gpu_G_loss += gan_power*(DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d)  + \
                                                 DLoss(gpu_pred_src_src_d2_ones, gpu_pred_src_src_d2))

                        if masked_training:
                            # Minimal src-src-bg rec with total_variation_mse to suppress random bright dots from gan
                            gpu_G_loss += 0.000001*nn.total_variation_mse(gpu_pred_src_src)
                            gpu_G_loss += 0.02*tf.reduce_mean(tf.square(gpu_pred_src_src_anti_masked-gpu_target_src_anti_masked),axis=[1,2,3] )

                    gpu_G_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]


            # Average losses and gradients, and create optimizer update ops
            with tf.device(f'/CPU:0'):
                pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)

            with tf.device (models_opt_device):
                src_loss = tf.concat(gpu_src_losses, 0)
                dst_loss = tf.concat(gpu_dst_losses, 0)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gvs))

                if self.options['true_face_power'] != 0:
                    D_loss_gv_op = self.D_code_opt.get_update_op (nn.average_gv_list(gpu_D_code_loss_gvs))

                if gan_power != 0:
                    src_D_src_dst_loss_gv_op = self.D_src_dst_opt.get_update_op (nn.average_gv_list(gpu_D_src_dst_loss_gvs) )


            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                              warped_dst, target_dst, target_dstm, target_dstm_em, ):
                s, d = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm:target_srcm,
                                                       self.target_srcm_em:target_srcm_em,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm:target_dstm,
                                                       self.target_dstm_em:target_dstm_em,
                                                       })[:2]
                return s, d
            self.src_dst_train = src_dst_train

            if self.options['true_face_power'] != 0:
                def D_train(warped_src, warped_dst):
                    nn.tf_sess.run ([D_loss_gv_op], feed_dict={self.warped_src: warped_src, self.warped_dst: warped_dst})
                self.D_train = D_train

            if gan_power != 0:
                def D_src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                                    warped_dst, target_dst, target_dstm, target_dstm_em, ):
                    nn.tf_sess.run ([src_D_src_dst_loss_gv_op], feed_dict={self.warped_src :warped_src,
                                                                           self.target_src :target_src,
                                                                           self.target_srcm:target_srcm,
                                                                           self.target_srcm_em:target_srcm_em,
                                                                           self.warped_dst :warped_dst,
                                                                           self.target_dst :target_dst,
                                                                           self.target_dstm:target_dstm,
                                                                           self.target_dstm_em:target_dstm_em})
                self.D_src_dst_train = D_src_dst_train


            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run ( [pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src:warped_src,
                                                    self.warped_dst:warped_dst})
            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device( nn.tf_default_device_name if len(devices) != 0 else f'/CPU:0'):
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
                if self.is_training and gan_power != 0 and model == self.D_src:
                    if self.gan_model_changed:
                        do_init = True

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()


        ###############

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None and not self.pretrain else None

            cpu_count = min(multiprocessing.cpu_count(), self.options['cpu_cap'])
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            fs_aug = None
            if ct_mode == 'fs-aug':
                fs_aug = 'fs-aug'

            

            channel_type = SampleProcessor.ChannelType.LAB_RAND_TRANSFORM if self.options['random_color'] else SampleProcessor.ChannelType.BGR

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp,
                                                 'random_downsample': self.options['random_downsample'],
                                                 'random_noise': self.options['random_noise'],
                                                 'random_blur': self.options['random_blur'],
                                                 'random_jpeg': self.options['random_jpeg'],
                                                 'random_shadow': random_shadow_src,
                                                 'random_hsv_shift_amount' : random_hsv_power,
                                                 'transform':True, 'channel_type' : channel_type, 'ct_mode': ct_mode,  
                                                 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution
                                                 },
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False,
                                                'transform':True, 'channel_type' : channel_type, 'ct_mode': ct_mode,
                                                'random_hsv_shift_amount' : random_hsv_power,
                                                'random_shadow': random_shadow_src,
                                                'face_type':self.face_type,
                                                'data_format':nn.data_format,
                                                'resolution': resolution
                                                },
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False,
                                                'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,
                                                'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,
                                                'face_type':self.face_type,
                                                'data_format':nn.data_format,
                                                'resolution': resolution
                                                },
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False,
                                                'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,
                                                'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE_EYES,
                                                'face_type':self.face_type,
                                                'data_format':nn.data_format,
                                                'resolution': resolution
                                                },
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=src_generators_count 
                    ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp,
                                                 'random_downsample': self.options['random_downsample'],
                                                 'random_noise': self.options['random_noise'],
                                                 'random_blur': self.options['random_blur'],
                                                 'random_jpeg': self.options['random_jpeg'],
                                                 'random_shadow': random_shadow_dst,
                                                 'transform':True, 'channel_type' : channel_type, 'ct_mode': fs_aug,
                                                 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False,
                                                'transform':True, 'channel_type' : channel_type, 'ct_mode': fs_aug,
                                                'random_shadow': random_shadow_dst,
                                                'random_hsv_shift_amount' : random_hsv_power,
                                                'face_type':self.face_type,
                                                'data_format':nn.data_format,
                                                'resolution': resolution
                                                },
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False,
                                                'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,
                                                'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,
                                                'face_type':self.face_type,
                                                'data_format':nn.data_format,
                                                'resolution': resolution
                                                },
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False,
                                                'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,
                                                'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE_EYES,
                                                'face_type':self.face_type,
                                                'data_format':nn.data_format,
                                                'resolution': resolution
                                                },
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=dst_generators_count
                    )
            ])

            if self.options['retraining_samples']:
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    def export_dfm (self):
        output_path=self.get_strpath_storage_for_file('model.dfm')

        io.log_info(f'Dumping .dfm to {output_path}')

        tf = nn.tf
        nn.set_data_format('NCHW')

        with tf.device (nn.tf_default_device_name):
            warped_dst = tf.placeholder (nn.floatx, (None, self.resolution, self.resolution, 3), name='in_face')
            warped_dst = tf.transpose(warped_dst, (0,3,1,2))


            if 'df' in self.archi_type:
                gpu_dst_code     = self.inter(self.encoder(warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            elif 'liae' in self.archi_type:
                gpu_dst_code = self.encoder (warped_dst)
                gpu_dst_inter_B_code = self.inter_B (gpu_dst_code)
                gpu_dst_inter_AB_code = self.inter_AB (gpu_dst_code)
                gpu_dst_code = tf.concat([gpu_dst_inter_B_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)
                gpu_src_dst_code = tf.concat([gpu_dst_inter_AB_code,gpu_dst_inter_AB_code], nn.conv2d_ch_axis)

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

            gpu_pred_src_dst = tf.transpose(gpu_pred_src_dst, (0,2,3,1))
            gpu_pred_dst_dstm = tf.transpose(gpu_pred_dst_dstm, (0,2,3,1))
            gpu_pred_src_dstm = tf.transpose(gpu_pred_src_dstm, (0,2,3,1))

        tf.identity(gpu_pred_dst_dstm, name='out_face_mask')
        tf.identity(gpu_pred_src_dst, name='out_celeb_face')
        tf.identity(gpu_pred_src_dstm, name='out_celeb_face_mask')

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            nn.tf_sess,
            tf.get_default_graph().as_graph_def(),
            ['out_face_mask','out_celeb_face','out_celeb_face_mask']
        )

        import tf2onnx
        with tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name='SAEHD',
                input_names=['in_face:0'],
                output_names=['out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0'],
                opset=12,
                output_path=output_path)

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

        ( (warped_src, target_src, target_srcm, target_srcm_em), \
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        if self.options['retraining_samples']:
            bs = self.get_batch_size()

            for i in range(bs):
                self.last_src_samples_loss.append (  (target_src[i], target_srcm[i], target_srcm_em[i], src_loss[i] )  )
                self.last_dst_samples_loss.append (  (target_dst[i], target_dstm[i], target_dstm_em[i], dst_loss[i] )  )

            if len(self.last_src_samples_loss) >= bs*16:
                src_samples_loss = sorted(self.last_src_samples_loss, key=operator.itemgetter(3), reverse=True)
                dst_samples_loss = sorted(self.last_dst_samples_loss, key=operator.itemgetter(3), reverse=True)

                target_src        = np.stack( [ x[0] for x in src_samples_loss[:bs] ] )
                target_srcm       = np.stack( [ x[1] for x in src_samples_loss[:bs] ] )
                target_srcm_em    = np.stack( [ x[2] for x in src_samples_loss[:bs] ] )

                target_dst        = np.stack( [ x[0] for x in dst_samples_loss[:bs] ] )
                target_dstm       = np.stack( [ x[1] for x in dst_samples_loss[:bs] ] )
                target_dstm_em = np.stack( [ x[2] for x in dst_samples_loss[:bs] ] )

                src_loss, dst_loss = self.src_dst_train (target_src, target_src, target_srcm, target_srcm_em, target_dst, target_dst, target_dstm, target_dstm_em)
                self.last_src_samples_loss = []
                self.last_dst_samples_loss = []

        if self.options['true_face_power'] != 0 and not self.pretrain:
            self.D_train (warped_src, warped_dst)

        if self.gan_power != 0:
            self.D_src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )
    #override
    def onGetPreview(self, samples, for_history=False, filenames=None):
        ( (warped_src, target_src, target_srcm, target_srcm_em),
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = samples

        S, D, SS, SSM, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        SW, DW = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([warped_src,warped_dst]) ]
        SSM, DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [SSM, DDM, SDM] ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(self.get_batch_size(), self.options['preview_samples'])

        if filenames is not None and len(filenames) > 0:
            for i in range(n_samples):
                S[i] = label_face_filename(S[i], filenames[0][i])
                D[i] = label_face_filename(D[i], filenames[1][i])

        if self.resolution <= 256 or self.options['force_full_preview'] == True:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD', np.concatenate (st, axis=0 )), ]

            wt = []
            for i in range(n_samples):
                ar = SW[i], SS[i], DW[i], DD[i], SD[i]
                wt.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD warped', np.concatenate (wt, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                SM = S[i]*target_srcm[i]
                DM = D[i]*target_dstm[i]
                if filenames is not None and len(filenames) > 0:
                    SM = label_face_filename(SM, filenames[0][i])
                    DM = label_face_filename(DM, filenames[1][i])
                ar = SM, SS[i]*SSM[i], DM, DD[i]*DDM[i], SD[i]*SD_mask
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

            wt = []
            for i in range(n_samples):
                ar = SW[i], SS[i]
                wt.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD warped src-src', np.concatenate (wt, axis=0 )), ]

            wt = []
            for i in range(n_samples):
                ar = DW[i], DD[i]
                wt.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD warped dst-dst', np.concatenate (wt, axis=0 )), ]

            wt = []
            for i in range(n_samples):
                ar = DW[i], SD[i]
                wt.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD warped pred', np.concatenate (wt, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                ar = S[i]*target_srcm[i], SS[i]*SSM[i]
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

    #override
    def get_config_schema_path(self):
        config_path = Path(__file__).parent.absolute() / Path("config_schema.json")
        return config_path

    #override
    def get_formatted_configuration_path(self):
        config_path = Path(__file__).parent.absolute() / Path("formatted_config.yaml")
        return config_path

Model = SAEHDModel
