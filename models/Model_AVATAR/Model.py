from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2
from nnlib import tf_dssim
from nnlib import DSSIMLossClass
from nnlib import conv
from nnlib import upscale

class Model(ModelBase):

    encoder64H5 = 'encoder64.h5'
    decoder64_srcH5 = 'decoder64_src.h5'
    decoder64_dstH5 = 'decoder64_dst.h5'
    encoder256H5 = 'encoder256.h5'    
    decoder256H5 = 'decoder256.h5'

    #override
    def onInitialize(self, **in_options):
        tf = self.tf
        keras = self.keras
        K = keras.backend
        
        self.set_vram_batch_requirements( {3.5:8,4:8,5:12,6:16,7:24,8:32,9:48} )
        if self.batch_size < 4:
            self.batch_size = 4   

        img_shape64, img_shape256, self.encoder64, self.decoder64_src, self.decoder64_dst, self.encoder256, self.decoder256 = self.Build()   
  
        if not self.is_first_run():
            self.encoder64.load_weights      (self.get_strpath_storage_for_file(self.encoder64H5))
            self.decoder64_src.load_weights  (self.get_strpath_storage_for_file(self.decoder64_srcH5))
            self.decoder64_dst.load_weights  (self.get_strpath_storage_for_file(self.decoder64_dstH5))
            self.encoder256.load_weights     (self.get_strpath_storage_for_file(self.encoder256H5))
            self.decoder256.load_weights (self.get_strpath_storage_for_file(self.decoder256H5))
            
        if self.is_training_mode:
            self.encoder64, self.decoder64_src, self.decoder64_dst, self.encoder256, self.decoder256 = self.to_multi_gpu_model_if_possible ( [self.encoder64, self.decoder64_src, self.decoder64_dst, self.encoder256, self.decoder256] )
        
        input_A_warped64 = keras.layers.Input(img_shape64)
        input_B_warped64 = keras.layers.Input(img_shape64)
        A_rec64 = self.decoder64_src(self.encoder64(input_A_warped64))
        B_rec64 = self.decoder64_dst(self.encoder64(input_B_warped64))
        self.ae64 = self.keras.models.Model([input_A_warped64, input_B_warped64], [A_rec64, B_rec64] )

        if self.is_training_mode:
            self.ae64, = self.to_multi_gpu_model_if_possible ( [self.ae64,] )

        self.ae64.compile(optimizer=self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999),
                        loss=[DSSIMLossClass(self.tf)(), DSSIMLossClass(self.tf)()] )
                         
        self.A64_view = K.function ([input_A_warped64], [A_rec64])
        self.B64_view = K.function ([input_B_warped64], [B_rec64])

        input_A_warped64 = keras.layers.Input(img_shape64)
        input_A_target256 = keras.layers.Input(img_shape256)
        A_rec256 = self.decoder256( self.encoder256(input_A_warped64)   )       
        
        input_B_warped64 = keras.layers.Input(img_shape64)
        BA_rec64 = self.decoder64_src( self.encoder64(input_B_warped64) )
        BA_rec256 = self.decoder256( self.encoder256(BA_rec64)  )       

        self.ae256 = self.keras.models.Model([input_A_warped64], [A_rec256] )

        if self.is_training_mode:
            self.ae256, = self.to_multi_gpu_model_if_possible ( [self.ae256,] )

        self.ae256.compile(optimizer=self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999),
                        loss=[DSSIMLossClass(self.tf)()])

        self.A256_view = K.function ([input_A_warped64], [A_rec256])        
        self.BA256_view = K.function ([input_B_warped64], [BA_rec256])
    
        if self.is_training_mode:
            from models import TrainingDataGenerator
            f = TrainingDataGenerator.SampleTypeFlags 
            self.set_training_data_generators ([            
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ 
                        [f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64],
                        [f.TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64],       
                        [f.TRANSFORMED | f.FULL_FACE | f.MODE_BGR, 256], 
                        [f.SOURCE | f.HALF_FACE | f.MODE_BGR, 64], 
                        [f.SOURCE | f.HALF_FACE | f.MODE_BGR, 256] ] ),
                        
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ 
                        [f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64],
                        [f.TRANSFORMED | f.HALF_FACE | f.MODE_BGR, 64],                        
                        [f.SOURCE | f.HALF_FACE | f.MODE_BGR, 64], 
                        [f.SOURCE | f.HALF_FACE | f.MODE_BGR, 256] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder64, self.get_strpath_storage_for_file(self.encoder64H5)],
                                 [self.decoder64_src, self.get_strpath_storage_for_file(self.decoder64_srcH5)],
                                 [self.decoder64_dst, self.get_strpath_storage_for_file(self.decoder64_dstH5)],
                                 [self.encoder256, self.get_strpath_storage_for_file(self.encoder256H5)],
                                 [self.decoder256, self.get_strpath_storage_for_file(self.decoder256H5)],
                                 ] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src64, target_src64, target_src256, target_src_source64, target_src_source256 = sample[0]
        warped_dst64, target_dst64, target_dst_source64, target_dst_source256 = sample[1]    
        
        loss64, loss_src64, loss_dst64 = self.ae64.train_on_batch ([warped_src64, warped_dst64], [target_src64, target_dst64])
        
        loss256 = self.ae256.train_on_batch ([warped_src64], [target_src256])   
        
        return ( ('loss64', loss64 ), ('loss256', loss256), )

    #override
    def onGetPreview(self, sample):
        sample_src64_source  = sample[0][3][0:4]
        sample_src256_source = sample[0][4][0:4]
        
        sample_dst64_source  = sample[1][2][0:4]
        sample_dst256_source = sample[1][3][0:4] 
        
        SRC64,  = self.A64_view ([sample_src64_source])
        DST64,  = self.B64_view ([sample_dst64_source])
        SRCDST64,  = self.A64_view ([sample_dst64_source])
        DSTSRC64,  = self.B64_view ([sample_src64_source])
        
        SRC_x1_256,      = self.A256_view ([sample_src64_source])
        DST_x2_256,      = self.BA256_view ([sample_dst64_source])
        
        b1 = np.concatenate ( (
                                np.concatenate ( (sample_src64_source[0], SRC64[0], sample_src64_source[1], SRC64[1], ), axis=1),
                                np.concatenate ( (sample_src64_source[1], SRC64[1], sample_src64_source[3], SRC64[3], ), axis=1),
                                np.concatenate ( (sample_dst64_source[0], DST64[0], sample_dst64_source[1], DST64[1], ), axis=1),
                                np.concatenate ( (sample_dst64_source[2], DST64[2], sample_dst64_source[3], DST64[3], ), axis=1),
                                ), axis=0 )
        
        b2 = np.concatenate ( (
                                np.concatenate ( (sample_src64_source[0], DSTSRC64[0], sample_src64_source[1], DSTSRC64[1], ), axis=1),
                                np.concatenate ( (sample_src64_source[2], DSTSRC64[2], sample_src64_source[3], DSTSRC64[3], ), axis=1),     
                                np.concatenate ( (sample_dst64_source[0], SRCDST64[0], sample_dst64_source[1], SRCDST64[1], ), axis=1),
                                np.concatenate ( (sample_dst64_source[2], SRCDST64[2], sample_dst64_source[3], SRCDST64[3], ), axis=1),
                                                           
                                ), axis=0 )
                                
        result = np.concatenate ( ( np.concatenate ( (b1, sample_src256_source[0], SRC_x1_256[0] ), axis=1 ),
                                    np.concatenate ( (b2, sample_dst256_source[0], DST_x2_256[0] ), axis=1 ),
                                   ), axis = 0 )

        return [ ('AVATAR', result ) ]

    def predictor_func (self, img):
        x, = self.BA256_view ([ np.expand_dims(img, 0) ])[0]
        return x
        
    #override
    def get_converter(self, **in_options):
        return ConverterAvatar(self.predictor_func, predictor_input_size=64, output_size=256, **in_options)
        
    def Build(self):
        keras, K = self.keras, self.keras.backend
        
        img_shape64  = (64,64,3)
        img_shape256  = (256,256,3)

        def Encoder(_input):
            x = _input
            x = self.keras.layers.convolutional.Conv2D(90, kernel_size=5, strides=1, padding='same')(x)
            x = self.keras.layers.convolutional.Conv2D(90, kernel_size=5, strides=1, padding='same')(x)
            x = self.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
            
            x = self.keras.layers.convolutional.Conv2D(180, kernel_size=3, strides=1, padding='same')(x)
            x = self.keras.layers.convolutional.Conv2D(180, kernel_size=3, strides=1, padding='same')(x)
            x = self.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
            
            x = self.keras.layers.convolutional.Conv2D(360, kernel_size=3, strides=1, padding='same')(x)
            x = self.keras.layers.convolutional.Conv2D(360, kernel_size=3, strides=1, padding='same')(x)
            x = self.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
            
            x = self.keras.layers.Dense (1024)(x)
            x = self.keras.layers.advanced_activations.LeakyReLU(0.1)(x)
            x = self.keras.layers.Dropout(0.5)(x)
            
            x = self.keras.layers.Dense (1024)(x)
            x = self.keras.layers.advanced_activations.LeakyReLU(0.1)(x)
            x = self.keras.layers.Dropout(0.5)(x)
            x = self.keras.layers.Flatten()(x)
            x = self.keras.layers.Dense (64)(x)
            
            return keras.models.Model (_input, x)
            
        encoder256 = Encoder( keras.layers.Input (img_shape64) )
        encoder64 = Encoder( keras.layers.Input (img_shape64) )

        def decoder256(encoder):
            decoder_input = keras.layers.Input ( K.int_shape(encoder.outputs[0])[1:] )
            x = decoder_input
            x = self.keras.layers.Dense(16 * 16 * 720)(x)
            x = keras.layers.Reshape ( (16, 16, 720) )(x)
            x = upscale(keras, x, 720)
            x = upscale(keras, x, 360)
            x = upscale(keras, x, 180)
            x = upscale(keras, x, 90)
            x = keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            return keras.models.Model(decoder_input, x)
        
        def decoder64(encoder):
            decoder_input = keras.layers.Input ( K.int_shape(encoder.outputs[0])[1:] )
            x = decoder_input
            x = self.keras.layers.Dense(8 * 8 * 720)(x)
            x = keras.layers.Reshape ( (8, 8, 720) )(x)
            x = upscale(keras, x, 360)
            x = upscale(keras, x, 180)
            x = upscale(keras, x, 90)
            x = keras.layers.convolutional.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            return keras.models.Model(decoder_input, x)
            
        return img_shape64, img_shape256, encoder64, decoder64(encoder64), decoder64(encoder64), encoder256, decoder256(encoder256)
        
from models import ConverterBase
from facelib import FaceType
from facelib import LandmarksProcessor
class ConverterAvatar(ConverterBase):

    #override
    def __init__(self,  predictor,
                        predictor_input_size=0, 
                        output_size=0,               
                        **in_options):
                        
        super().__init__(predictor)
         
        self.predictor_input_size = predictor_input_size
        self.output_size = output_size   
  
    #override
    def get_mode(self):
        return ConverterBase.MODE_IMAGE
        
    #override
    def dummy_predict(self):
        self.predictor ( np.zeros ( (self.predictor_input_size, self.predictor_input_size,3), dtype=np.float32) )
        
    #override
    def convert_image (self, img_bgr, img_face_landmarks, debug):
        img_size = img_bgr.shape[1], img_bgr.shape[0]
        
        face_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.predictor_input_size, face_type=FaceType.HALF )
        predictor_input_bgr = cv2.warpAffine( img_bgr, face_mat, (self.predictor_input_size, self.predictor_input_size), flags=cv2.INTER_LANCZOS4 )

        predicted_bgr = self.predictor ( predictor_input_bgr )

        output = cv2.resize ( predicted_bgr, (self.output_size, self.output_size), cv2.INTER_LANCZOS4 )
        if debug:
            return (img_bgr,output,)
        return output  