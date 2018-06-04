from models import ModelBase
from models import TrainingDataType
import numpy as np
import cv2
from nnlib import tf_dssim
from nnlib import conv
from nnlib import upscale

class Model(ModelBase):

    encoder64H5 = 'encoder64.h5'
    decoder64_srcH5 = 'decoder64_src.h5'
    decoder64_dstH5 = 'decoder64_dst.h5'
    encoder128H5 = 'encoder128.h5'    
    decoder128_srcH5 = 'decoder128_src.h5'

    #override
    def onInitialize(self, **in_options):
        tf = self.tf
        keras = self.keras
        K = keras.backend
        
        self.set_vram_batch_requirements( {4:8,5:16,6:20,7:24,8:32,9:48} )
                
        self.encoder64, self.decoder64_src, self.decoder64_dst, self.encoder128, self.decoder128_src = self.BuildAE()   
        img_shape64      = (64,64,1)
        img_shape128   = (256,256,3)
        
        if not self.is_first_run():
            self.encoder64.load_weights      (self.get_strpath_storage_for_file(self.encoder64H5))
            self.decoder64_src.load_weights  (self.get_strpath_storage_for_file(self.decoder64_srcH5))
            self.decoder64_dst.load_weights  (self.get_strpath_storage_for_file(self.decoder64_dstH5))
            self.encoder128.load_weights     (self.get_strpath_storage_for_file(self.encoder128H5))
            self.decoder128_src.load_weights (self.get_strpath_storage_for_file(self.decoder128_srcH5))
            
        if self.is_training_mode:
            self.encoder64, self.decoder64_src, self.decoder64_dst, self.encoder128, self.decoder128_src = self.to_multi_gpu_model_if_possible ( [self.encoder64, self.decoder64_src, self.decoder64_dst, self.encoder128, self.decoder128_src] )
        
        input_src_64         = keras.layers.Input(img_shape64)
        input_src_target64   = keras.layers.Input(img_shape64)
        input_src_target128  = keras.layers.Input(img_shape128)
        input_dst_64         = keras.layers.Input(img_shape64)
        input_dst_target64   = keras.layers.Input(img_shape64)

        src_code64 = self.encoder64(input_src_64)
        dst_code64 = self.encoder64(input_dst_64)
        
        rec_src64 = self.decoder64_src(src_code64)
        rec_dst64 = self.decoder64_dst(dst_code64)        
        
        src64_loss = tf_dssim(tf, input_src_target64, rec_src64)
        dst64_loss = tf_dssim(tf, input_dst_target64, rec_dst64)
        total64_loss = src64_loss + dst64_loss        

        self.ed64_train = K.function ([input_src_64, input_src_target64, input_dst_64, input_dst_target64],[K.mean(total64_loss)],
                                       self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(total64_loss, self.encoder64.trainable_weights + self.decoder64_src.trainable_weights + self.decoder64_dst.trainable_weights)
                                     )
                                     
        src_code128 = self.encoder128(input_src_64)  
        rec_src128 = self.decoder128_src(src_code128)
        src128_loss = tf_dssim(tf, input_src_target128, rec_src128)
                                     
        self.ed128_train = K.function ([input_src_64, input_src_target128],[K.mean(src128_loss)],
                                       self.keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(src128_loss, self.encoder128.trainable_weights + self.decoder128_src.trainable_weights)
                                     )                             
                       
        src_code128 = self.encoder128(rec_src64)  
        rec_src128 = self.decoder128_src(src_code128)
        
        self.src128_view = K.function ([input_src_64], [rec_src128])
    
        if self.is_training_mode:
            from models import TrainingDataGenerator
            f = TrainingDataGenerator.SampleTypeFlags 
            self.set_training_data_generators ([            
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ 
                        [f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_G, 64],
                        [f.TRANSFORMED | f.HALF_FACE | f.MODE_G, 64],       
                        [f.TRANSFORMED | f.FULL_FACE | f.MODE_BGR, 256], 
                        [f.SOURCE | f.HALF_FACE | f.MODE_G, 64], 
                        [f.SOURCE | f.HALF_FACE | f.MODE_GGG, 256] ] ),
                        
                    TrainingDataGenerator(TrainingDataType.FACE, self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, output_sample_types=[ 
                        [f.WARPED_TRANSFORMED | f.HALF_FACE | f.MODE_G, 64],
                        [f.TRANSFORMED | f.HALF_FACE | f.MODE_G, 64],                        
                        [f.SOURCE | f.HALF_FACE | f.MODE_G, 64], 
                        [f.SOURCE | f.HALF_FACE | f.MODE_GGG, 256] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder64, self.get_strpath_storage_for_file(self.encoder64H5)],
                                 [self.decoder64_src, self.get_strpath_storage_for_file(self.decoder64_srcH5)],
                                 [self.decoder64_dst, self.get_strpath_storage_for_file(self.decoder64_dstH5)],
                                 [self.encoder128, self.get_strpath_storage_for_file(self.encoder128H5)],
                                 [self.decoder128_src, self.get_strpath_storage_for_file(self.decoder128_srcH5)],
                                 ] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src64, target_src64, target_src128, target_src_source64_G, target_src_source128_GGG = sample[0]
        warped_dst64, target_dst64, target_dst_source64_G, target_dst_source128_GGG = sample[1]    
        
        loss64,  = self.ed64_train  ([warped_src64, target_src64, warped_dst64, target_dst64])   
        loss256, = self.ed128_train ([warped_src64, target_src128])   
        
        return ( ('loss64', loss64), ('loss256', loss256) )

    #override
    def onGetPreview(self, sample):
        n_samples = 4
        test_B    = sample[1][2][0:n_samples]
        test_B128 = sample[1][3][0:n_samples] 
        
        BB,      = self.src128_view ([test_B])

        st = []
        for i in range(n_samples // 2):
            st.append ( np.concatenate ( (
                test_B128[i*2+0], BB[i*2+0], test_B128[i*2+1], BB[i*2+1],
                ), axis=1) )
        return [ ('AVATAR', np.concatenate ( st, axis=0 ) ) ]

    def predictor_func (self, img):
        x, = self.src128_view ([ np.expand_dims(img, 0) ])[0]
        return x
        
    #override
    def get_converter(self, **in_options):
        return ConverterAvatar(self.predictor_func, predictor_input_size=64, output_size=256, **in_options)
        
    def BuildAE(self):
        keras, K = self.keras, self.keras.backend

        def Encoder(_input):
            x = keras.layers.convolutional.Conv2D(90, kernel_size=5, strides=1, padding='same')(_input)
            x = keras.layers.convolutional.Conv2D(90, kernel_size=5, strides=1, padding='same')(x)
            x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
            
            x = keras.layers.convolutional.Conv2D(180, kernel_size=3, strides=1, padding='same')(x)
            x = keras.layers.convolutional.Conv2D(180, kernel_size=3, strides=1, padding='same')(x)
            x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
            
            x = keras.layers.convolutional.Conv2D(360, kernel_size=3, strides=1, padding='same')(x)
            x = keras.layers.convolutional.Conv2D(360, kernel_size=3, strides=1, padding='same')(x)
            x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
            
            x = keras.layers.Dense (1024)(x)
            x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
            x = keras.layers.Dropout(0.5)(x)
            
            x = keras.layers.Dense (1024)(x)
            x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense (64)(x)            
            return keras.models.Model (_input, x)
            
        encoder128 = Encoder( keras.layers.Input ( (64, 64, 1) ) )
        encoder64 = Encoder( keras.layers.Input ( (64, 64, 1) ) )

        def decoder128_3(encoder):
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
        
        def decoder64_1(encoder):
            decoder_input = keras.layers.Input ( K.int_shape(encoder.outputs[0])[1:] )
            x = decoder_input
            x = self.keras.layers.Dense(8 * 8 * 720)(x)
            x = keras.layers.Reshape ( (8,8,720) )(x)
            x = upscale(keras, x, 360)
            x = upscale(keras, x, 180)
            x = upscale(keras, x, 90)
            x = keras.layers.convolutional.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(x)
            return keras.models.Model(decoder_input, x)
            
        return encoder64, decoder64_1(encoder64), decoder64_1(encoder64), encoder128, decoder128_3(encoder128)
        
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
        self.predictor ( np.zeros ( (self.predictor_input_size, self.predictor_input_size,1), dtype=np.float32) )
        
    #override
    def convert_image (self, img_bgr, img_face_landmarks, debug):
        img_size = img_bgr.shape[1], img_bgr.shape[0]
        
        face_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.predictor_input_size, face_type=FaceType.HALF )
        predictor_input_bgr = cv2.warpAffine( img_bgr, face_mat, (self.predictor_input_size, self.predictor_input_size), flags=cv2.INTER_LANCZOS4 )
        predictor_input_g   = np.expand_dims(cv2.cvtColor(predictor_input_bgr, cv2.COLOR_BGR2GRAY),-1)
        
        predicted_bgr = self.predictor ( predictor_input_g )

        output = cv2.resize ( predicted_bgr, (self.output_size, self.output_size), cv2.INTER_LANCZOS4 )
        if debug:
            return (img_bgr,output,)
        return output  