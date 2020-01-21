import multiprocessing
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

class QModel(ModelBase):
    #override
    def on_initialize(self):
        nn.initialize()
        tf = nn.tf

        conv_kernel_initializer = nn.initializers.ca
        
        class Downscale(nn.ModelBase):
            def __init__(self, in_ch, out_ch, kernel_size=5, dilations=1, subpixel=True, use_activator=True, *kwargs ):
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.kernel_size = kernel_size
                self.dilations = dilations
                self.subpixel = subpixel
                self.use_activator = use_activator
                super().__init__(*kwargs)

            def on_build(self, *args, **kwargs ):
                self.conv1 = nn.Conv2D( self.in_ch,
                                          self.out_ch // (4 if self.subpixel else 1),
                                          kernel_size=self.kernel_size,
                                          strides=1 if self.subpixel else 2,
                                          padding='SAME', dilations=self.dilations, kernel_initializer=conv_kernel_initializer )

            def forward(self, x):
                x = self.conv1(x)

                if self.subpixel:
                    x = tf.nn.space_to_depth(x, 2)

                if self.use_activator:
                    x = tf.nn.leaky_relu(x, 0.2)
                return x

            def get_out_ch(self):
                return (self.out_ch // 4) * 4

        class DownscaleBlock(nn.ModelBase):
            def on_build(self, in_ch, ch, n_downscales, kernel_size, dilations=1, subpixel=True):
                self.downs = []

                last_ch = in_ch
                for i in range(n_downscales):
                    cur_ch = ch*( min(2**i, 8)  )
                    self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size, dilations=dilations, subpixel=subpixel) )
                    last_ch = self.downs[-1].get_out_ch()

            def forward(self, inp):
                x = inp
                for down in self.downs:
                    x = down(x)
                return x
       
        class Upscale(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, x):
                x = self.conv1(x)
                x = tf.nn.leaky_relu(x, 0.2)
                x = tf.nn.depth_to_space(x, 2)
                return x
                
        class UpdownResidualBlock(nn.ModelBase):
            def on_build(self, ch, inner_ch, kernel_size=3 ):
                self.up   = Upscale (ch, inner_ch, kernel_size=kernel_size)
                self.res  = ResidualBlock (inner_ch, kernel_size=kernel_size)
                self.down = Downscale (inner_ch, ch, kernel_size=kernel_size, use_activator=False)

            def forward(self, inp):
                x = self.up(inp)
                x = upx = self.res(x)
                x = self.down(x)
                x = x + inp
                x = tf.nn.leaky_relu(x, 0.2)
                return x, upx
                
        class ResidualBlock(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, inp):
                x = self.conv1(inp)
                x = tf.nn.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = inp + x
                x = tf.nn.leaky_relu(x, 0.2)
                return x

        class Encoder(nn.ModelBase):
            def on_build(self, in_ch, e_ch):
                self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5)
            def forward(self, inp):
                return nn.tf_flatten(self.down1(inp))

        class Inter(nn.ModelBase):
            def __init__(self, in_ch, lowest_dense_res, ae_ch, ae_out_ch, d_ch, **kwargs):
                self.in_ch, self.lowest_dense_res, self.ae_ch, self.ae_out_ch, self.d_ch = in_ch, lowest_dense_res, ae_ch, ae_out_ch, d_ch
                super().__init__(**kwargs)

            def on_build(self):
                in_ch, lowest_dense_res, ae_ch, ae_out_ch, d_ch = self.in_ch, self.lowest_dense_res, self.ae_ch, self.ae_out_ch, self.d_ch

                self.dense1 = nn.Dense( in_ch, ae_ch, kernel_initializer=tf.initializers.orthogonal )
                self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch, maxout_features=2, kernel_initializer=tf.initializers.orthogonal )
                self.upscale1 = Upscale(ae_out_ch, d_ch*8)
                self.res1 = ResidualBlock(d_ch*8)

            def forward(self, inp):
                x = self.dense1(inp)
                x = self.dense2(x)
                x = tf.reshape (x, (-1, lowest_dense_res, lowest_dense_res, self.ae_out_ch))
                x = self.upscale1(x)
                x = self.res1(x)
                return x

            def get_out_ch(self):
                return self.ae_out_ch

        class Decoder(nn.ModelBase):
            def on_build(self, in_ch, d_ch):        
                self.upscale1 = Upscale(in_ch, d_ch*4)
                
                self.res1     = UpdownResidualBlock(d_ch*4, d_ch*2)                
                self.upscale2 = Upscale(d_ch*4, d_ch*2)                
                self.res2     = UpdownResidualBlock(d_ch*2, d_ch)                
                self.upscale3 = Upscale(d_ch*2, d_ch*1)
                self.res3     = UpdownResidualBlock(d_ch, d_ch//2)

                self.upscalem1 = Upscale(in_ch, d_ch)
                self.upscalem2 = Upscale(d_ch, d_ch//2)
                self.upscalem3 = Upscale(d_ch//2, d_ch//2)

                self.out_conv = nn.Conv2D( d_ch*1, 3, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)
                self.out_convm = nn.Conv2D( d_ch//2, 1, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, inp):
                z = inp               
                                                
                x = self.upscale1(z)             
                x, upx = self.res1(x)
                
                x = self.upscale2(x)
                x = tf.nn.leaky_relu(x + upx, 0.2)                    
                x, upx = self.res2(x)
                
                x = self.upscale3(x)
                x = tf.nn.leaky_relu(x + upx, 0.2)                    
                x, upx = self.res3(x)  
                    
                """
                x = self.upscale1 (z)
                x = self.res1     (x)
                x = self.upscale2 (x)
                x = self.res2     (x)
                x = self.upscale3 (x)
                x = self.res3     (x)
                """

                y = self.upscalem1 (z)
                y = self.upscalem2 (y)
                y = self.upscalem3 (y)

                return tf.nn.sigmoid(self.out_conv(x)), \
                       tf.nn.sigmoid(self.out_convm(y))

        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        resolution = self.resolution = 96
        ae_dims = 128
        e_dims = 128
        d_dims = 64
        self.pretrain = True
        self.pretrain_just_disabled = False
        
        masked_training = True

        models_opt_on_gpu = len(devices) == 1 and devices[0].total_mem_gb >= 4
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_nc = 3
        output_nc = 3
        bgr_shape = (resolution, resolution, output_nc)
        mask_shape = (resolution, resolution, 1)
        lowest_dense_res = resolution // 16

        self.model_filename_list = []


        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (tf.float32, (None,)+bgr_shape)
            self.warped_dst = tf.placeholder (tf.float32, (None,)+bgr_shape)

            self.target_src = tf.placeholder (tf.float32, (None,)+bgr_shape)
            self.target_dst = tf.placeholder (tf.float32, (None,)+bgr_shape)

            self.target_srcm = tf.placeholder (tf.float32, (None,)+mask_shape)
            self.target_dstm = tf.placeholder (tf.float32, (None,)+mask_shape)

        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(in_ch=input_nc, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.compute_output_shape ( (tf.float32, (None,resolution,resolution,input_nc)))[-1]

            self.inter = Inter (in_ch=encoder_out_ch, lowest_dense_res=lowest_dense_res, ae_ch=ae_dims, ae_out_ch=ae_dims, d_ch=d_dims, name='inter')
            inter_out_ch = self.inter.compute_output_shape ( (tf.float32, (None,encoder_out_ch)))[-1]

            self.decoder_src = Decoder(in_ch=inter_out_ch, d_ch=d_dims, name='decoder_src')
            self.decoder_dst = Decoder(in_ch=inter_out_ch, d_ch=d_dims, name='decoder_dst')

            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                self.src_dst_trainable_weights = self.encoder.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                
                # Initialize optimizers
                self.src_dst_opt = nn.TFRMSpropOptimizer(lr=2e-4, lr_dropout=0.3, name='src_dst_opt')
                self.src_dst_opt.initialize_variables(self.src_dst_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu )
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, 4 // gpu_count)
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
            gpu_src_dst_loss_gvs = []

            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        gpu_warped_src   = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst   = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src   = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst   = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm  = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_dstm  = self.target_dstm[batch_slice,:,:,:]

                    # process model tensors                    
                    gpu_src_code     = self.inter(self.encoder(gpu_warped_src))
                    gpu_dst_code     = self.inter(self.encoder(gpu_warped_dst))
                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)
                    
                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)
                    
                    gpu_target_srcm_blur = nn.tf_gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_dstm_blur = nn.tf_gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_target_srcmasked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss =  tf.reduce_mean ( 10*nn.tf_dssim(gpu_target_srcmasked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_srcmasked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    gpu_src_loss += tf.reduce_mean ( tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    gpu_dst_loss  = tf.reduce_mean ( 10*nn.tf_dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean ( tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_src_dst_loss = gpu_src_loss + gpu_dst_loss
                    gpu_src_dst_loss_gvs += [ nn.tf_gradients ( gpu_src_dst_loss, self.src_dst_trainable_weights ) ]


            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                if gpu_count == 1:
                    pred_src_src = gpu_pred_src_src_list[0]
                    pred_dst_dst = gpu_pred_dst_dst_list[0]
                    pred_src_dst = gpu_pred_src_dst_list[0]
                    pred_src_srcm = gpu_pred_src_srcm_list[0]
                    pred_dst_dstm = gpu_pred_dst_dstm_list[0]
                    pred_src_dstm = gpu_pred_src_dstm_list[0]
                    
                    src_loss = gpu_src_losses[0]
                    dst_loss = gpu_dst_losses[0]
                    src_dst_loss_gv = gpu_src_dst_loss_gvs[0]
                else:
                    pred_src_src = tf.concat(gpu_pred_src_src_list, 0)
                    pred_dst_dst = tf.concat(gpu_pred_dst_dst_list, 0)
                    pred_src_dst = tf.concat(gpu_pred_src_dst_list, 0)
                    pred_src_srcm = tf.concat(gpu_pred_src_srcm_list, 0)
                    pred_dst_dstm = tf.concat(gpu_pred_dst_dstm_list, 0)
                    pred_src_dstm = tf.concat(gpu_pred_src_dstm_list, 0)
                    
                    src_loss = nn.tf_average_tensor_list(gpu_src_losses)
                    dst_loss = nn.tf_average_tensor_list(gpu_dst_losses)
                    src_dst_loss_gv = nn.tf_average_gv_list (gpu_src_dst_loss_gvs)

                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (src_dst_loss_gv)

            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm, \
                              warped_dst, target_dst, target_dstm):
                s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm:target_srcm,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm:target_dstm,
                                                       })
                s = np.mean(s)
                d = np.mean(d)
                return s, d
            self.src_dst_train = src_dst_train

            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src:warped_src,
                                                    self.warped_dst:warped_dst})

            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
                gpu_dst_code     = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            def AE_merge( warped_dst):
                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

            self.AE_merge = AE_merge
            
        
        
        
        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = self.is_first_run()
                        
            if self.pretrain_just_disabled:
                if model == self.inter:
                    do_init = True

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init and self.pretrained_model_path is not None:                
                pretrained_filepath = self.pretrained_model_path / filename
                if pretrained_filepath.exists():
                    do_init = not model.load_weights(pretrained_filepath)
                    
            if do_init:
                model.init_weights()

        # initializing sample generators

        if self.is_training:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = multiprocessing.cpu_count()

            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count - src_generators_count

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':resolution, },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution': resolution, },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution } ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution} ],
                        generators_count=dst_generators_count )
                             ])
                             
            self.last_samples = None

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )


    #override
    def onTrainOneIter(self):
        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = self.last_samples 
            src_loss, dst_loss = self.src_dst_train (target_src, target_src, target_srcm, 
                                                     target_dst, target_dst, target_dstm)
        else:
            samples = self.last_samples = self.generate_next_samples()
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = samples

            src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm, 
                                                     warped_dst, target_dst, target_dstm)
        
        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, samples):
        n_samples = min(4, self.get_batch_size() )

        ( (warped_src, target_src, target_srcm),
          (warped_dst, target_dst, target_dstm) ) = \
                [ [sample[0:n_samples] for sample in sample_list ]
                                                 for sample_list in samples ]

        S, D, SS, DD, DDM, SD, SDM = [ np.clip(x, 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96', np.concatenate (st, axis=0 )), ]

        st_m = []
        for i in range(n_samples):
            ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
            st_m.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96 masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):

        bgr, mask_dst_dstm, mask_src_dstm = self.AE_merge (face[np.newaxis,...])
        mask = mask_dst_dstm[0] * mask_src_dstm[0]
        return bgr[0], mask[...,0]

    #override
    def get_MergerConfig(self):
        face_type = FaceType.FULL

        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=face_type,
                                     default_mode = 'overlay',
                                     clip_hborder_mask_per=0.0625 if (face_type != FaceType.HALF) else 0,
                                    )

Model = QModel
