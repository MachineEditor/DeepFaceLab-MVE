import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from facelib import FANSegmentator
from samples import *
from interact import interact as io

class Model(ModelBase):

    
    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {1.5:4} )

        self.resolution = 256
        self.face_type = FaceType.FULL
        
        self.fan_seg = FANSegmentator(self.resolution, 
                                      FaceType.toString(self.face_type), 
                                      load_weights=not self.is_first_run(),
                                      weights_file_root=self.get_model_root_path() )

        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            f_type = f.FACE_ALIGN_FULL #if self.face_type == FaceType.FULL else f.FACE_ALIGN_HALF
            
            self.set_training_data_generators ([    
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ), 
                            output_sample_types=[ [f.TRANSFORMED | f_type | f.MODE_BGR, self.resolution],
                                                  [f.TRANSFORMED | f_type | f.MODE_M | f.FACE_MASK_FULL, self.resolution]
                                                ]),
                                                
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ), 
                            output_sample_types=[ [f.TRANSFORMED | f_type | f.MODE_BGR, self.resolution]
                                                ])
                                               ])
                
    #override
    def onSave(self):        
        self.fan_seg.save_weights()
        
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        target_src, target_src_mask = generators_samples[0]

        loss = self.fan_seg.train_on_batch( [target_src], [target_src_mask] )

        return ( ('loss', loss), )
        
    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][0][0:4] #first 4 samples
        test_B   = sample[1][0][0:4] #first 4 samples
        
        mAA = self.fan_seg.extract_from_bgr([test_A])
        mBB = self.fan_seg.extract_from_bgr([test_B])
        mAA = np.repeat ( mAA, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                mAA[i],
                test_A[i,:,:,0:3]*mAA[i],
                ), axis=1) )
                
        st2 = []
        for i in range(0, len(test_B)):
            st2.append ( np.concatenate ( (
                test_B[i,:,:,0:3],
                mBB[i],
                test_B[i,:,:,0:3]*mBB[i],
                ), axis=1) )
                
        return [ ('FANSegmentator', np.concatenate ( st, axis=0 ) ),
                 ('never seen', np.concatenate ( st2, axis=0 ) ),
                 ]

    def predictor_func (self, face):
        
        face_64_bgr = face[...,0:3]
        face_64_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.src_view ( [ np.expand_dims(face_64_bgr,0) ] )
        x, mx = x[0], mx[0]     
        
        return np.concatenate ( (x,mx), -1 )

    #override
    def get_converter(self):
        from converters import ConverterMasked
        return ConverterMasked(self.predictor_func,
                               predictor_input_size=64, 
                               output_size=64, 
                               face_type=FaceType.HALF, 
                               base_erode_mask_modifier=100,
                               base_blur_mask_modifier=100)
        
   