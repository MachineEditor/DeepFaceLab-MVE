from functools import partial

import numpy as np

import mathlib
from facelib import FaceType
from interact import interact as io
from models import ModelBase
from nnlib import nnlib
from samplelib import *


class Quick96Model(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                            ask_enable_autobackup=False,
                            ask_write_preview_history=False,
                            ask_target_iter=False,
                            ask_batch_size=False,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)                 
                 
    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements({1.5:2,2:4})#,3:4,4:8})

        resolution = self.resolution = 96
        
        class CommonModel(object):
            def downscale (self, dim, kernel_size=5, dilation_rate=1):
                def func(x):
                    return SubpixelDownscaler()(ELU()(Conv2D(dim // 4, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x)))
                return func

            def upscale (self, dim, size=(2,2)):
                def func(x):
                    return SubpixelUpscaler(size=size)(ELU()(Conv2D(dim * np.prod(size) , kernel_size=3, strides=1, padding='same')(x)))
                return func

            def ResidualBlock(self, dim):
                def func(inp):
                    x = Conv2D(dim, kernel_size=3, padding='same')(inp)
                    x = LeakyReLU(0.2)(x)
                    x = Conv2D(dim, kernel_size=3, padding='same')(x)
                    x = Add()([x, inp])  
                    x = LeakyReLU(0.2)(x)  
                    return x
                return func

        class QModel(CommonModel):
            def __init__(self, resolution, ae_dims, e_dims, d_dims):
                super().__init__()
                bgr_shape = (resolution, resolution, 3)
                mask_shape = (resolution, resolution, 1)
                lowest_dense_res = resolution // 16

                def enc_flow():
                    def func(inp):
                        x = self.downscale(e_dims, 3, 1 )(inp)                        
                        x = self.downscale(e_dims*2, 3, 1 )(x)
                        x = self.downscale(e_dims*4, 3, 1 )(x)
                        x0 = self.downscale(e_dims*8, 3, 1 )(x)         
                                         
                        x = self.downscale(e_dims, 3, 2 )(inp)                        
                        x = self.downscale(e_dims*2, 3, 2 )(x)
                        x = self.downscale(e_dims*4, 3, 2 )(x)
                        x1 = self.downscale(e_dims*8, 3, 2 )(x)     
                                             
                        x = Concatenate()([x0,x1])        
                                        
                        x = DenseMaxout(ae_dims, kernel_initializer='orthogonal')(Flatten()(x))
                        x = DenseMaxout(lowest_dense_res * lowest_dense_res * ae_dims, kernel_initializer='orthogonal')(x)
                        x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
                        
                        x = self.ResidualBlock(ae_dims)(x)
                        x = self.upscale(d_dims*8)(x)
                        x = self.ResidualBlock(d_dims*8)(x)
                        return x
                    return func

                def dec_flow():
                    def func(inp):
                        x = self.upscale(d_dims*4)(inp)
                        x = self.ResidualBlock(d_dims*4)(x)
                        x = self.upscale(d_dims*2)(x)
                        x = self.ResidualBlock(d_dims*2)(x)
                        x = self.upscale(d_dims)(x)
                        x = self.ResidualBlock(d_dims)(x)
                        
                        y = self.upscale(d_dims)(inp)
                        y = self.upscale(d_dims//2)(y)
                        y = self.upscale(d_dims//4)(y)
                        
                        return Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x), \
                               Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)

                    return func

                self.encoder = modelify(enc_flow()) ( Input(bgr_shape) )

                sh = K.int_shape( self.encoder.outputs[0] )[1:]
                self.decoder_src = modelify(dec_flow()) ( Input(sh) )
                self.decoder_dst = modelify(dec_flow()) ( Input(sh) )

                self.src_trainable_weights = self.encoder.trainable_weights + self.decoder_src.trainable_weights
                self.dst_trainable_weights = self.encoder.trainable_weights + self.decoder_dst.trainable_weights

                self.warped_src, self.warped_dst = Input(bgr_shape), Input(bgr_shape)
                self.target_src, self.target_dst = Input(bgr_shape), Input(bgr_shape)
                self.target_srcm, self.target_dstm = Input(mask_shape), Input(mask_shape)
                                
                self.src_code = self.encoder(self.warped_src)                            
                self.dst_code = self.encoder(self.warped_dst)    

                self.pred_src_src, self.pred_src_srcm = self.decoder_src(self.src_code)
                self.pred_dst_dst, self.pred_dst_dstm = self.decoder_dst(self.dst_code)
                self.pred_src_dst, self.pred_src_dstm = self.decoder_src(self.dst_code)

            def get_model_filename_list(self, exclude_for_pretrain=False):
                ar = []
                if not exclude_for_pretrain:
                    ar += [ [self.encoder, 'encoder.h5'] ]
                ar += [  [self.decoder_src, 'decoder_src.h5'],
                         [self.decoder_dst, 'decoder_dst.h5']  ]
                         
                return ar
                
        self.model = QModel (resolution, 128, 64, 64)

        loaded, not_loaded = [], self.model.get_model_filename_list()
        if not self.is_first_run():
            loaded, not_loaded = self.load_weights_safe(not_loaded)

        CA_models = [ model for model, _ in not_loaded ]

        self.CA_conv_weights_list = []
        for model in CA_models:
            for layer in model.layers:
                if type(layer) == keras.layers.Conv2D:
                    self.CA_conv_weights_list += [layer.weights[0]] #- is Conv2D kernel_weights

        if self.is_training_mode:
            self.src_dst_opt      = RMSprop(lr=2e-4)
            self.src_dst_mask_opt = RMSprop(lr=2e-4)
                
            target_src_masked = self.model.target_src*self.model.target_srcm
            target_dst_masked = self.model.target_dst*self.model.target_dstm

            pred_src_src_masked = self.model.pred_src_src*self.model.target_srcm
            pred_dst_dst_masked = self.model.pred_dst_dst*self.model.target_dstm
            
            src_loss =  K.mean ( 10*dssim(kernel_size=int(resolution/11.6),max_value=2.0)( target_src_masked+1, pred_src_src_masked+1) )
            src_loss += K.mean ( 10*K.square( target_src_masked - pred_src_src_masked ) )
            src_loss += K.mean(K.square(self.model.target_srcm-self.model.pred_src_srcm))

            dst_loss = K.mean( 10*dssim(kernel_size=int(resolution/11.6),max_value=2.0)(target_dst_masked+1, pred_dst_dst_masked+1) )
            dst_loss += K.mean( 10*K.square( target_dst_masked - pred_dst_dst_masked ) )
            dst_loss += K.mean(K.square(self.model.target_dstm-self.model.pred_dst_dstm))

            self.src_train = K.function ([self.model.warped_src, self.model.target_src, self.model.target_srcm], [src_loss], self.src_dst_opt.get_updates( src_loss, self.model.src_trainable_weights) )
            self.dst_train = K.function ([self.model.warped_dst, self.model.target_dst, self.model.target_dstm], [dst_loss], self.src_dst_opt.get_updates( dst_loss, self.model.dst_trainable_weights) )
            self.AE_view = K.function ([self.model.warped_src, self.model.warped_dst], [self.model.pred_src_src, self.model.pred_dst_dst, self.model.pred_dst_dstm, self.model.pred_src_dst, self.model.pred_src_dstm])
        else:
            self.AE_convert = K.function ([self.model.warped_dst],[ self.model.pred_src_dst, self.model.pred_dst_dstm, self.model.pred_src_dstm ])

        if self.is_training_mode:
            t = SampleProcessor.Types

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution': resolution, 'normalize_tanh':True },
                                                {'types' : (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution': resolution, 'normalize_tanh':True },
                                                {'types' : (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_M), 'resolution': resolution } ]
                                              ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, ),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution': resolution, 'normalize_tanh':True },
                                                {'types' : (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution': resolution, 'normalize_tanh':True },
                                                {'types' : (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_M), 'resolution': resolution} ])
                             ])
            self.counter = 0
    
    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list ()

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list() )

    #override
    def on_success_train_one_iter(self):        
        if len(self.CA_conv_weights_list) != 0:
            exec(nnlib.import_all(), locals(), globals())
            CAInitializerMP ( self.CA_conv_weights_list )
            self.CA_conv_weights_list = []

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, target_src, target_srcm = generators_samples[0]
        warped_dst, target_dst, target_dstm = generators_samples[1]
        
        self.counter += 1
        if self.counter % 3 == 0:
            src_loss, = self.src_train ([warped_src, target_src, target_srcm])
            dst_loss, = self.dst_train ([warped_dst, target_dst, target_dstm])
        else:
            src_loss, = self.src_train ([target_src, target_src, target_srcm])
            dst_loss, = self.dst_train ([target_dst, target_dst, target_dstm])

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, sample):
        test_S   = sample[0][1][0:4] #first 4 samples
        test_S_m = sample[0][2][0:4] #first 4 samples
        test_D   = sample[1][1][0:4]
        test_D_m = sample[1][2][0:4]

        S, D, SS, DD, DDM, SD, SDM = [test_S,test_D] + self.AE_view ([test_S, test_D])        
        S, D, SS, DD, SD, = [ np.clip(x/2+0.5, 0.0, 1.0) for x in [S, D, SS, DD, SD] ]        
        DDM, SDM, = [ np.clip( np.repeat (x, (3,), -1), 0, 1) for x in [DDM, SDM] ]

        result = []
        st = []
        for i in range(len(test_S)):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96', np.concatenate (st, axis=0 )), ]
        
        st_m = []
        for i in range(len(test_S)):
            ar = S[i]*test_S_m[i], SS[i], D[i]*test_D_m[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
            st_m.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96 masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.AE_convert ([ np.zeros ( (1, self.resolution, self.resolution, 3), dtype=np.float32 ) ])
        else:
            face = face * 2 - 1
            bgr, mask_dst_dstm, mask_src_dstm = self.AE_convert ([face[np.newaxis,...]])
            bgr = bgr /2 + 0.5
            mask = mask_dst_dstm[0] * mask_src_dstm[0]
            return bgr[0], mask[...,0]

    #override
    def get_ConverterConfig(self):
        import converters
        return self.predictor_func, (self.resolution, self.resolution, 3), converters.ConverterConfigMasked(face_type=FaceType.FULL,
                                     default_mode='seamless', clip_hborder_mask_per=0.0625)

Model = Quick96Model
