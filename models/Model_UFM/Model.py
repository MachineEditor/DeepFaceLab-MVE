import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *
from utils.console_utils import *

#U-net Face Morpher
class UFMModel(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'
    decoder_srcmH5 = 'decoder_srcm.h5'
    decoder_dstmH5 = 'decoder_dstm.h5'
    
    #override
    def onInitializeOptions(self, is_first_run, ask_for_session_options):
        default_resolution = 128
        default_filters = 64
        default_match_style = True
        default_face_type = 'f'
        
        if is_first_run: 
            #first run
            self.options['resolution'] = input_int("Resolution (valid: 64,128,256, skip:128) : ", default_resolution, [64,128,256])
            self.options['filters'] = np.clip ( input_int("Number of U-net filters (valid: 32-128, skip:64) : ", default_filters), 32, 128 )
            self.options['match_style'] = input_bool ("Match style? (y/n skip:y) : ", default_match_style)            
            self.options['face_type'] = input_str ("Half or Full face? (h/f, skip:f) : ", default_face_type, ['h','f'])
            
        else: 
            #not first run
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['filters'] = self.options.get('filters', default_filters)
            self.options['match_style'] = self.options.get('match_style', default_match_style)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
    
    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())

        self.set_vram_batch_requirements({2:1,3:2,4:6,5:8,6:16,7:24,8:32})
        
        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        
        filters = self.options['filters']
        
        if resolution == 64:
            lowest_dense = 512
        elif resolution == 128:
            lowest_dense = 512
        elif resolution == 256:
            lowest_dense = 256
            
        self.encoder = modelify(UFMModel.EncFlow (ngf=filters, lowest_dense=lowest_dense)) (Input(bgr_shape))
        
        dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
        
        self.decoder_src = modelify(UFMModel.DecFlow (bgr_shape[2], ngf=filters)) (dec_Inputs)
        self.decoder_dst = modelify(UFMModel.DecFlow (bgr_shape[2], ngf=filters)) (dec_Inputs)
        
        self.decoder_srcm = modelify(UFMModel.DecFlow (mask_shape[2], ngf=filters//2)) (dec_Inputs)
        self.decoder_dstm = modelify(UFMModel.DecFlow (mask_shape[2], ngf=filters//2)) (dec_Inputs)

        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))
            self.decoder_srcm.load_weights (self.get_strpath_storage_for_file(self.decoder_srcmH5))
            self.decoder_dstm.load_weights (self.get_strpath_storage_for_file(self.decoder_dstmH5))
 
        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)
        
        warped_src_code = self.encoder (warped_src)
        pred_src_src = self.decoder_src(warped_src_code)
        pred_src_srcm = self.decoder_srcm(warped_src_code)
        
        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)
        
        warped_dst_code = self.encoder (warped_dst)
        pred_dst_dst = self.decoder_dst(warped_dst_code)
        pred_dst_dstm = self.decoder_dstm(warped_dst_code)
        
        pred_src_dst = self.decoder_src(warped_dst_code)
        pred_src_dstm = self.decoder_srcm(warped_dst_code)
        
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
        
        target_src_masked = target_src_sigm*target_srcm_sigm
        
        target_dst_masked = target_dst_sigm * target_dstm_sigm
        target_dst_anti_masked = target_dst_sigm * target_dstm_anti_sigm
        
        pred_src_src_masked = pred_src_src_sigm * target_srcm_sigm        
        pred_dst_dst_masked = pred_dst_dst_sigm * target_dstm_sigm
        
        pred_src_dst_target_dst_masked = pred_src_dst_sigm * target_dstm_sigm
        pred_src_dst_target_dst_anti_masked = pred_src_dst_sigm * target_dstm_anti_sigm
        

        src_loss = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked, pred_src_src_masked )) ) 
        if self.options['match_style']:
            src_loss += tf_style_loss(gaussian_blur_radius=resolution // 8, loss_weight=0.015)(pred_src_dst_target_dst_masked, target_dst_masked) 
            src_loss += 0.05 * K.mean( tf_dssim(2.0)( pred_src_dst_target_dst_anti_masked, target_dst_anti_masked ))

        self.src_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(src_loss, self.encoder.trainable_weights + self.decoder_src.trainable_weights) )
                                    
        dst_loss = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked, pred_dst_dst_masked )) )        
        self.dst_train = K.function ([warped_dst, target_dst, target_dstm],[dst_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(dst_loss, self.encoder.trainable_weights + self.decoder_dst.trainable_weights) )
                                    
        
        src_mask_loss = K.mean(K.square(target_srcm-pred_src_srcm))                                       
        self.src_mask_train = K.function ([warped_src, target_srcm],[src_mask_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(src_mask_loss, self.encoder.trainable_weights + self.decoder_srcm.trainable_weights) )
        
        dst_mask_loss = K.mean(K.square(target_dstm-pred_dst_dstm))                                       
        self.dst_mask_train = K.function ([warped_dst, target_dstm],[dst_mask_loss],
                                    Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(dst_mask_loss, self.encoder.trainable_weights + self.decoder_dstm.trainable_weights) )
                                    
        self.AE_view = K.function ([warped_src, warped_dst],[pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm])
        self.AE_convert = K.function ([warped_dst],[pred_src_dst, pred_src_dstm])
        
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            
            face_type = f.FACE_ALIGN_FULL if self.options['face_type'] == 'f' else f.FACE_ALIGN_HALF
            
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution] ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                                [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)],
                                [self.decoder_srcm, self.get_strpath_storage_for_file(self.decoder_srcmH5)],
                                [self.decoder_dstm, self.get_strpath_storage_for_file(self.decoder_dstmH5)]
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
            
        return [ ('U-net Face Morpher', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        
        face = face * 2.0 - 1.0
        
        face_128_bgr = face[...,0:3]
 
        x, mx = [ (x[0] + 1.0) / 2.0 for x in self.AE_convert ( [ np.expand_dims(face_128_bgr,0) ] ) ]
        
        if self.options['match_style']:
            res = self.options['resolution']
            s = int( res * 0.96875 )
            mx = np.pad ( np.ones ( (s,s) ), (res-s) // 2 , mode='constant')
            mx = np.expand_dims(mx, -1)
        
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked

        if self.options['match_style']:
            base_erode_mask_modifier = 50
            base_blur_mask_modifier = 50
        else:
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
    def EncFlow(ngf=64, num_downs=4, lowest_dense=512):
        exec (nnlib.import_all(), locals(), globals())
        
        use_bias = True
        def XNormalization(x):
            return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
            
        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def func(input):     
            x = input
            
            result = []
            for i in range(num_downs):
                x = LeakyReLU(0.1)(XNormalization(Conv2D( min(ngf* (2**i), ngf*8) , 5, 2, 'same')(x)))
                
                if i == 3:
                    x_shape = K.int_shape(x)[1:]
                    x = Reshape(x_shape)(Dense( np.prod(x_shape) )(Dense(lowest_dense)(Flatten()(x))))
                result += [x]
               
            return result
        return func
        
    @staticmethod
    def DecFlow(output_nc, ngf=64, activation='tanh'):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = True
        def XNormalization(x):
            return InstanceNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)
            
        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def func(input):
            input_len = len(input)
            
            x = input[input_len-1]
            for i in range(input_len-1, -1, -1):          
                x = SubpixelUpscaler()( LeakyReLU(0.1)(XNormalization(Conv2D( min(ngf* (2**i) *4, ngf*8 *4 ), 3, 1, 'same')(x))) )
                if i != 0:
                    x = Concatenate(axis=3)([ input[i-1] , x])
        
            return Conv2D(output_nc, 3, 1, 'same', activation=activation)(x)
        return func

Model = UFMModel