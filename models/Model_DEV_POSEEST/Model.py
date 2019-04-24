import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from facelib import PoseEstimator
from samplelib import *
from interact import interact as io
import imagelib
 
class Model(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                            ask_write_preview_history=False, 
                            ask_target_iter=False,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)
    
    #override
    def onInitializeOptions(self, is_first_run, ask_override):            
        default_face_type = 'f'
        if is_first_run:
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)
     
    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {4:32} )

        self.resolution = 128
        self.face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        
        self.pose_est = PoseEstimator(self.resolution, 
                                      FaceType.toString(self.face_type), 
                                      load_weights=not self.is_first_run(),
                                      weights_file_root=self.get_model_root_path(),
                                      training=True)

        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            face_type = f.FACE_TYPE_FULL if self.options['face_type'] == 'f' else f.FACE_TYPE_HALF

        self.set_training_data_generators ([    
                SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, generators_count=4,
                        sample_process_options=SampleProcessor.Options( rotation_range=[0,0], motion_blur = [25, 1] ), #random_flip=True,
                        output_sample_types=[ [f.TRANSFORMED | face_type | f.MODE_BGR_SHUFFLE | f.OPT_APPLY_MOTION_BLUR, self.resolution ],
                                              [f.PITCH_YAW_ROLL],
                                            ]),
                                            
                SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, generators_count=4,
                        sample_process_options=SampleProcessor.Options( rotation_range=[0,0] ), #random_flip=True,
                        output_sample_types=[ [f.TRANSFORMED | face_type | f.MODE_BGR_SHUFFLE, self.resolution ],
                                               [f.PITCH_YAW_ROLL],
                                            ])
                                           ])
                
    #override
    def onSave(self):        
        self.pose_est.save_weights()
        
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        target_src, pitch_yaw_roll = generators_samples[0]
    
        pitch_loss,yaw_loss,roll_loss = self.pose_est.train_on_batch( target_src, pitch_yaw_roll )

        return ( ('pitch_loss', pitch_loss), ('yaw_loss', yaw_loss), ('roll_loss', roll_loss) )
        
    #override
    def onGetPreview(self, generators_samples):
        test_src     = generators_samples[0][0][0:4] #first 4 samples
        test_pyr_src = generators_samples[0][1][0:4]
        test_dst     = generators_samples[1][0][0:4]
        test_pyr_dst = generators_samples[1][1][0:4]

        h,w,c = self.resolution,self.resolution,3
        h_line = 13
   
        result = []
        for name, img, pyr in [ ['training data', test_src, test_pyr_src],  \
                                ['evaluating data',test_dst, test_pyr_dst] ]:
            pyr_pred = self.pose_est.extract(img)

            hor_imgs = []
            for i in range(len(img)):
                img_info = np.ones ( (h,w,c) ) * 0.1
                lines = ["%s" % ( str(pyr[i]) ),
                         "%s" % ( str(pyr_pred[i]) ) ]
                        
                lines_count = len(lines)
                for ln in range(lines_count):                
                    img_info[ ln*h_line:(ln+1)*h_line, 0:w] += \
                        imagelib.get_text_image (  (h_line,w,c), lines[ln], color=[0.8]*c )
                        
                hor_imgs.append ( np.concatenate ( (
                    img[i,:,:,0:3],
                    img_info
                    ), axis=1) )
                    
            
            result += [ (name, np.concatenate (hor_imgs, axis=0)) ] 
            
        return result