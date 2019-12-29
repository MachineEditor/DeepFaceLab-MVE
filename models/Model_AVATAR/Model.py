from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *

from facelib import PoseEstimator

class AVATARModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_random_flip=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        if is_first_run:
            #avatar_type = io.input_int("Avatar type ( 0:source, 1:head, 2:full_face ?:help skip:1) : ", 1, [0,1,2],
            #                           help_message="Training target for the model. Source is direct untouched images. Full_face or head are centered nose unaligned faces.")
            #avatar_type = {0:'source',
            #               1:'head',
            #               2:'full_face'}[avatar_type]

            self.options['avatar_type'] = 'head'
        else:
            self.options['avatar_type'] = self.options.get('avatar_type', 'head')

        if is_first_run or ask_override:
            def_stage = self.options.get('stage', 1)
            self.options['stage'] = io.input_int("Stage (0, 1, 2 ?:help skip:%d) : " % def_stage, def_stage, [0,1,2], help_message="Train first stage, then second. Tune batch size to maximum possible for both stages.")
        else:
            self.options['stage'] = self.options.get('stage', 1)

    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({6:4})

        resolution = self.resolution = 224
        avatar_type = self.options['avatar_type']
        stage = self.stage = self.options['stage']
        df_res = self.df_res = 128
        df_bgr_shape = (df_res, df_res, 3)
        df_mask_shape = (df_res, df_res, 1)
        res_bgr_shape = (resolution, resolution, 3)
        res_bgr_t_shape = (resolution, resolution, 9)

        self.enc = modelify(AVATARModel.EncFlow())( [Input(df_bgr_shape),] )

        self.decA64 = modelify(AVATARModel.DecFlow()) ( [ Input(K.int_shape(self.enc.outputs[0])[1:]) ] )
        self.decB64 = modelify(AVATARModel.DecFlow()) ( [ Input(K.int_shape(self.enc.outputs[0])[1:]) ] )
        self.D = modelify(AVATARModel.Discriminator() ) (Input(df_bgr_shape))
        self.C = modelify(AVATARModel.ResNet (9, n_blocks=6, ngf=128, use_dropout=False))( Input(res_bgr_t_shape))

        if self.is_first_run():
            conv_weights_list = []
            for model, _ in self.get_model_filename_list():
                for layer in model.layers:
                    if type(layer) == keras.layers.Conv2D:
                        conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights
            CAInitializerMP ( conv_weights_list )

        if not self.is_first_run():
            self.load_weights_safe( self.get_model_filename_list() )

        def DLoss(labels,logits):
            return K.mean(K.binary_crossentropy(labels,logits))

        warped_A64 = Input(df_bgr_shape)
        real_A64 = Input(df_bgr_shape)
        real_A64m = Input(df_mask_shape)

        real_B64_t0 = Input(df_bgr_shape)
        real_B64_t1 = Input(df_bgr_shape)
        real_B64_t2 = Input(df_bgr_shape)

        real_A64_t0 = Input(df_bgr_shape)
        real_A64m_t0 = Input(df_mask_shape)
        real_A_t0 = Input(res_bgr_shape)
        real_A64_t1 = Input(df_bgr_shape)
        real_A64m_t1 = Input(df_mask_shape)
        real_A_t1 = Input(res_bgr_shape)
        real_A64_t2 = Input(df_bgr_shape)
        real_A64m_t2 = Input(df_mask_shape)
        real_A_t2 = Input(res_bgr_shape)

        warped_B64 = Input(df_bgr_shape)
        real_B64 = Input(df_bgr_shape)
        real_B64m = Input(df_mask_shape)

        warped_A_code = self.enc (warped_A64)
        warped_B_code = self.enc (warped_B64)

        rec_A64 = self.decA64(warped_A_code)
        rec_B64 = self.decB64(warped_B_code)
        rec_AB64 = self.decA64(warped_B_code)

        def Lambda_grey_mask (x,m):
            return Lambda (lambda x: x[0]*m+(1-m)*0.5, output_shape= K.int_shape(x)[1:3] + (3,)) ([x, m])

        def Lambda_gray_pad(x):
            a = np.ones((resolution,resolution,3))*0.5
            pad = ( resolution - df_res ) // 2
            a[pad:-pad:,pad:-pad:,:] = 0

            return Lambda ( lambda x: K.spatial_2d_padding(x, padding=((pad, pad), (pad, pad)) ) + K.constant(a, dtype=K.floatx() ),
                     output_shape=(resolution,resolution,3) ) (x)

        def Lambda_concat ( x ):
            c = sum ( [ K.int_shape(l)[-1] for l in x ] )
            return Lambda ( lambda x: K.concatenate (x, axis=-1), output_shape=K.int_shape(x[0])[1:3] + (c,) ) (x)

        def Lambda_Cto3t(x):
            return Lambda ( lambda x: x[...,0:3], output_shape= K.int_shape(x)[1:3] + (3,) ) (x), \
                   Lambda ( lambda x: x[...,3:6], output_shape= K.int_shape(x)[1:3] + (3,) ) (x), \
                   Lambda ( lambda x: x[...,6:9], output_shape= K.int_shape(x)[1:3] + (3,) ) (x)

        real_A64_d = self.D( Lambda_grey_mask(real_A64, real_A64m) )

        real_A64_d_ones = K.ones_like(real_A64_d)
        fake_A64_d = self.D(rec_AB64)
        fake_A64_d_ones = K.ones_like(fake_A64_d)
        fake_A64_d_zeros = K.zeros_like(fake_A64_d)

        rec_AB_t0 = Lambda_gray_pad( self.decA64 (self.enc (real_B64_t0)) )
        rec_AB_t1 = Lambda_gray_pad( self.decA64 (self.enc (real_B64_t1)) )
        rec_AB_t2 = Lambda_gray_pad( self.decA64 (self.enc (real_B64_t2)) )

        C_in_A_t0 = Lambda_gray_pad( Lambda_grey_mask (real_A64_t0, real_A64m_t0) )
        C_in_A_t1 = Lambda_gray_pad( Lambda_grey_mask (real_A64_t1, real_A64m_t1) )
        C_in_A_t2 = Lambda_gray_pad( Lambda_grey_mask (real_A64_t2, real_A64m_t2) )

        rec_C_A_t0, rec_C_A_t1, rec_C_A_t2 = Lambda_Cto3t ( self.C ( Lambda_concat ( [C_in_A_t0, C_in_A_t1, C_in_A_t2]) ) )
        rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 = Lambda_Cto3t( self.C ( Lambda_concat ( [rec_AB_t0, rec_AB_t1, rec_AB_t2]) ) )

        #real_A_t012_d = self.CD ( K.concatenate ( [real_A_t0, real_A_t1,real_A_t2], axis=-1)  )
        #real_A_t012_d_ones = K.ones_like(real_A_t012_d)
        #rec_C_AB_t012_d = self.CD ( K.concatenate ( [rec_C_AB_t0,rec_C_AB_t1, rec_C_AB_t2], axis=-1) )
        #rec_C_AB_t012_d_ones = K.ones_like(rec_C_AB_t012_d)
        #rec_C_AB_t012_d_zeros = K.zeros_like(rec_C_AB_t012_d)

        self.G64_view = K.function([warped_A64, warped_B64],[rec_A64, rec_B64, rec_AB64])
        self.G_view = K.function([real_A64_t0, real_A64m_t0, real_A64_t1, real_A64m_t1, real_A64_t2, real_A64m_t2, real_B64_t0, real_B64_t1, real_B64_t2], [rec_C_A_t0, rec_C_A_t1, rec_C_A_t2, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2])

        if self.is_training_mode:
            loss_AB64 = K.mean(10 * dssim(kernel_size=int(df_res/11.6),max_value=1.0) ( rec_A64, real_A64*real_A64m + (1-real_A64m)*0.5) ) + \
                        K.mean(10 * dssim(kernel_size=int(df_res/11.6),max_value=1.0) ( rec_B64, real_B64*real_B64m + (1-real_B64m)*0.5) ) + 0.1*DLoss(fake_A64_d_ones, fake_A64_d )

            weights_AB64 = self.enc.trainable_weights + self.decA64.trainable_weights + self.decB64.trainable_weights

            loss_C = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t0, rec_C_A_t0 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t1, rec_C_A_t1 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t2, rec_C_A_t2 ) )
                     #0.1*DLoss(rec_C_AB_t012_d_ones, rec_C_AB_t012_d )

            weights_C = self.C.trainable_weights

            loss_D = (DLoss(real_A64_d_ones, real_A64_d ) + \
                      DLoss(fake_A64_d_zeros, fake_A64_d ) ) * 0.5

            #loss_CD = ( DLoss(real_A_t012_d_ones, real_A_t012_d) + \
            #            DLoss(rec_C_AB_t012_d_zeros, rec_C_AB_t012_d) ) * 0.5
            #
            #weights_CD = self.CD.trainable_weights

            def opt(lr=5e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2 if 'tensorflow' in self.device_config.backend else 0 )

            self.AB64_train = K.function ([warped_A64, real_A64, real_A64m, warped_B64, real_B64, real_B64m], [loss_AB64], opt().get_updates(loss_AB64, weights_AB64) )
            self.C_train = K.function ([real_A64_t0, real_A64m_t0, real_A_t0,
                                        real_A64_t1, real_A64m_t1, real_A_t1,
                                        real_A64_t2, real_A64m_t2, real_A_t2,
                                        real_B64_t0, real_B64_t1,  real_B64_t2],[ loss_C ], opt().get_updates(loss_C, weights_C) )

            self.D_train = K.function ([warped_A64, real_A64, real_A64m, warped_B64, real_B64, real_B64m],[loss_D], opt().get_updates(loss_D, self.D.trainable_weights) )


            #self.CD_train = K.function ([real_A64_t0, real_A64m_t0, real_A_t0,
            #                             real_A64_t1, real_A64m_t1, real_A_t1,
            #                             real_A64_t2, real_A64m_t2, real_A_t2,
            #                             real_B64_t0, real_B64_t1,  real_B64_t2 ],[ loss_CD ], opt().get_updates(loss_CD, weights_CD) )

            ###########
            t = SampleProcessor.Types

            training_target = {'source' : t.NONE,
                               'full_face' : t.FACE_TYPE_FULL_NO_ALIGN,
                               'head' : t.FACE_TYPE_HEAD_NO_ALIGN}[avatar_type]

            generators = [
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_M), 'resolution':df_res}
                                            ] ),
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_M), 'resolution':df_res}
                                            ] ),

                    SampleGeneratorFaceTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[{'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},#IMG_WARPED_TRANSFORMED
                                             {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_M), 'resolution':df_res},
                                             {'types': (t.IMG_SOURCE, training_target, t.MODE_BGR), 'resolution':resolution},
                                            ] ),

                    SampleGeneratorFaceTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[{'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ALIGN, t.MODE_BGR), 'resolution':df_res},
                                             {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                            ] ),
                   ]

            if self.stage == 1:
                generators[2].set_active(False)
                generators[3].set_active(False)
            elif self.stage == 2:
                generators[0].set_active(False)
                generators[1].set_active(False)

            self.set_training_data_generators (generators)
        else:
            self.G_convert = K.function([real_B64_t0, real_B64_t1, real_B64_t2],[rec_C_AB_t1])

    #override , return [ [model, filename],... ]  list
    def get_model_filename_list(self):
        return [   [self.enc, 'enc.h5'],
                    [self.decA64, 'decA64.h5'],
                    [self.decB64, 'decB64.h5'],
                    [self.C, 'C.h5'],
                    [self.D, 'D.h5'],
                    #[self.CD, 'CD.h5'],
               ]

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list() )

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src64, src64, src64m = generators_samples[0]
        warped_dst64, dst64, dst64m = generators_samples[1]

        real_A64_t0, real_A64m_t0, real_A_t0, real_A64_t1, real_A64m_t1, real_A_t1, real_A64_t2, real_A64m_t2, real_A_t2 = generators_samples[2]
        real_B64_t0, _, real_B64_t1, _, real_B64_t2, _ = generators_samples[3]

        if self.stage == 0 or self.stage == 1:
            loss,   = self.AB64_train ( [warped_src64, src64, src64m, warped_dst64, dst64, dst64m] )
            loss_D, = self.D_train  ( [warped_src64, src64, src64m, warped_dst64, dst64, dst64m] )
            if self.stage != 0:
                loss_C = loss_CD = 0

        if self.stage == 0 or self.stage == 2:
            loss_C1, = self.C_train ( [real_A64_t0, real_A64m_t0, real_A_t0,
                                       real_A64_t1, real_A64m_t1, real_A_t1,
                                       real_A64_t2, real_A64m_t2, real_A_t2,
                                       real_B64_t0, real_B64_t1, real_B64_t2] )

            loss_C2, = self.C_train ( [real_A64_t2, real_A64m_t2, real_A_t2,
                                       real_A64_t1, real_A64m_t1, real_A_t1,
                                       real_A64_t0, real_A64m_t0, real_A_t0,
                                       real_B64_t0, real_B64_t1, real_B64_t2] )

            #loss_CD1, = self.CD_train ( [real_A64_t0, real_A64m_t0, real_A_t0,
            #                            real_A64_t1, real_A64m_t1, real_A_t1,
            #                            real_A64_t2, real_A64m_t2, real_A_t2,
            #                            real_B64_t0, real_B64_t1, real_B64_t2] )
            #
            #loss_CD2, = self.CD_train ( [real_A64_t2, real_A64m_t2, real_A_t2,
            #                             real_A64_t1, real_A64m_t1, real_A_t1,
            #                             real_A64_t0, real_A64m_t0, real_A_t0,
            #                             real_B64_t0, real_B64_t1, real_B64_t2] )

            loss_C = (loss_C1 + loss_C2) / 2
            #loss_CD = (loss_CD1 + loss_CD2) / 2
            if self.stage != 0:
                loss = loss_D = 0

        return ( ('loss', loss), ('D', loss_D), ('C', loss_C), ) #('CD', loss_CD) )

    #override
    def onGetPreview(self, sample):
        test_A064w  = sample[0][0][0:4]
        test_A064r  = sample[0][1][0:4]
        test_A064m  = sample[0][2][0:4]

        test_B064w  = sample[1][0][0:4]
        test_B064r  = sample[1][1][0:4]
        test_B064m  = sample[1][2][0:4]

        t_src64_0  = sample[2][0][0:4]
        t_src64m_0 = sample[2][1][0:4]
        t_src_0    = sample[2][2][0:4]
        t_src64_1  = sample[2][3][0:4]
        t_src64m_1 = sample[2][4][0:4]
        t_src_1    = sample[2][5][0:4]
        t_src64_2  = sample[2][6][0:4]
        t_src64m_2 = sample[2][7][0:4]
        t_src_2    = sample[2][8][0:4]

        t_dst64_0 = sample[3][0][0:4]
        t_dst_0   = sample[3][1][0:4]
        t_dst64_1 = sample[3][2][0:4]
        t_dst_1   = sample[3][3][0:4]
        t_dst64_2 = sample[3][4][0:4]
        t_dst_2   = sample[3][5][0:4]

        G64_view_result = self.G64_view ([test_A064r, test_B064r])
        test_A064r, test_B064r, rec_A64, rec_B64, rec_AB64 = [ x[0] for x in ([test_A064r, test_B064r] + G64_view_result)  ]

        sample64x4 = np.concatenate ([ np.concatenate ( [rec_B64, rec_A64], axis=1 ),
                                       np.concatenate ( [test_B064r, rec_AB64], axis=1) ], axis=0 )

        sample64x4 = cv2.resize (sample64x4, (self.resolution, self.resolution) )

        G_view_result = self.G_view([t_src64_0, t_src64m_0, t_src64_1, t_src64m_1, t_src64_2, t_src64m_2, t_dst64_0, t_dst64_1, t_dst64_2 ])

        t_dst_0, t_dst_1, t_dst_2, rec_C_A_t0, rec_C_A_t1, rec_C_A_t2, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 = [ x[0] for x in ([t_dst_0, t_dst_1, t_dst_2, ] + G_view_result)  ]

        c1 = np.concatenate ( (sample64x4, rec_C_A_t0, t_dst_0, rec_C_AB_t0 ), axis=1 )
        c2 = np.concatenate ( (sample64x4, rec_C_A_t1, t_dst_1, rec_C_AB_t1 ), axis=1 )
        c3 = np.concatenate ( (sample64x4, rec_C_A_t2, t_dst_2, rec_C_AB_t2 ), axis=1 )

        r = np.concatenate ( [c1,c2,c3], axis=0 )

        return [ ('AVATAR', r ) ]

    def predictor_func (self, prev_imgs=None, img=None, next_imgs=None, dummy_predict=False):
        if dummy_predict:
            z = np.zeros ( (1, self.df_res, self.df_res, 3), dtype=np.float32 )
            self.G_convert ([z,z,z])
        else:
            feed = [ prev_imgs[-1][np.newaxis,...], img[np.newaxis,...], next_imgs[0][np.newaxis,...] ]
            x = self.G_convert (feed)[0]
            return np.clip ( x[0], 0, 1)

    #override
    def get_ConverterConfig(self):
        import converters
        return self.predictor_func, (self.df_res, self.df_res, 3), converters.ConverterConfigFaceAvatar(temporal_face_count=1)

    @staticmethod
    def Discriminator(ndf=128):
        exec (nnlib.import_all(), locals(), globals())

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = Conv2D( ndf, 4, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
            x = LeakyReLU(0.2)(x)

            x = Conv2D( ndf*2, 4, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
            x = InstanceNormalization (axis=-1)(x)
            x = LeakyReLU(0.2)(x)

            x = Conv2D( ndf*4, 4, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
            x = InstanceNormalization (axis=-1)(x)
            x = LeakyReLU(0.2)(x)

            x = Conv2D( ndf*8, 4, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
            x = InstanceNormalization (axis=-1)(x)
            x = LeakyReLU(0.2)(x)

            return Conv2D( 1, 4, strides=1, padding='valid', activation='sigmoid')( ZeroPadding2D(3)(x) )
        return func

    @staticmethod
    def EncFlow():
        exec (nnlib.import_all(), locals(), globals())

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)( Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func


        def func(input):
            x, = input
            b,h,w,c = K.int_shape(x)

            dim_res = w // 16

            x = downscale(64)(x)
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)

            x = Dense(512)(Flatten()(x))
            x = Dense(dim_res * dim_res * 512)(x)
            x = Reshape((dim_res, dim_res, 512))(x)
            x = upscale(512)(x)
            return x

        return func

    @staticmethod
    def DecFlow(output_nc=3, **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func
        
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, strides=1, padding='same', activation='sigmoid')(x)
            return func
            
        def func(input):
            x = input[0]

            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            return to_bgr(output_nc) (x)

        return func
   
    @staticmethod
    def ResNet(output_nc, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        def func(input):
            def ResnetBlock(dim, use_dropout=False):
                def func(input):
                    x = input

                    x = Conv2D(dim, 3, strides=1, padding='same')(x)
                    x = InstanceNormalization (axis=-1)(x)
                    x = ReLU()(x)

                    if use_dropout:
                        x = Dropout(0.5)(x)

                    x = Conv2D(dim, 3, strides=1, padding='same')(x)
                    x = InstanceNormalization (axis=-1)(x)
                    x = ReLU()(x)
                    return Add()([x,input])
                return func

            x = input

            x = ReLU()(InstanceNormalization (axis=-1)(Conv2D(ngf, 7, strides=1, padding='same')(x)))

            x = ReLU()(InstanceNormalization (axis=-1)(Conv2D(ngf*2, 3, strides=2, padding='same')(x)))
            x = ReLU()(InstanceNormalization (axis=-1)(Conv2D(ngf*4, 3, strides=2, padding='same')(x)))

            x = ReLU()(InstanceNormalization (axis=-1)(Conv2D(ngf*4, 3, strides=2, padding='same')(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)

            x = ReLU()(InstanceNormalization (axis=-1)(Conv2DTranspose(ngf*4, 3, strides=2, padding='same')(x)))

            x = ReLU()(InstanceNormalization (axis=-1)(Conv2DTranspose(ngf*2, 3, strides=2, padding='same')(x)))
            x = ReLU()(InstanceNormalization (axis=-1)(Conv2DTranspose(ngf  , 3, strides=2, padding='same')(x)))

            x = Conv2D(output_nc, 7, strides=1, activation='sigmoid', padding='same')(x)

            return x

        return func

Model = AVATARModel