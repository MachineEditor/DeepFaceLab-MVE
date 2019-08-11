import time

import cv2
import numpy as np

from facelib import FaceType, LandmarksProcessor
from joblib import SubprocessFunctionCaller
from utils.pickle_utils import AntiPickler

from .Converter import Converter

class ConverterAvatar(Converter):

    #override
    def __init__(self,  predictor_func,
                        predictor_input_size=0):

        super().__init__(predictor_func, Converter.TYPE_FACE_AVATAR)

        self.predictor_input_size = predictor_input_size
        
        #dummy predict and sleep, tensorflow caching kernels. If remove it, conversion speed will be x2 slower
        predictor_func ( np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ),
                         np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ),
                         np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ) )
        time.sleep(2)

        predictor_func_host, predictor_func = SubprocessFunctionCaller.make_pair(predictor_func)
        self.predictor_func_host = AntiPickler(predictor_func_host)
        self.predictor_func = predictor_func

    #overridable
    def on_host_tick(self):
        self.predictor_func_host.obj.process_messages()
        
    #override
    def cli_convert_face (self, f0, f0_lmrk, f1, f1_lmrk, f2, f2_lmrk, debug, **kwargs):
        if debug:
            debugs = []
            
        inp_size = self.predictor_input_size
          
        f0_mat = LandmarksProcessor.get_transform_mat (f0_lmrk, inp_size, face_type=FaceType.FULL_NO_ALIGN)
        f1_mat = LandmarksProcessor.get_transform_mat (f1_lmrk, inp_size, face_type=FaceType.FULL_NO_ALIGN)
        f2_mat = LandmarksProcessor.get_transform_mat (f2_lmrk, inp_size, face_type=FaceType.FULL_NO_ALIGN)
        
        inp_f0 = cv2.warpAffine( f0, f0_mat, (inp_size, inp_size), flags=cv2.INTER_CUBIC )
        inp_f1 = cv2.warpAffine( f1, f1_mat, (inp_size, inp_size), flags=cv2.INTER_CUBIC )
        inp_f2 = cv2.warpAffine( f2, f2_mat, (inp_size, inp_size), flags=cv2.INTER_CUBIC )

        prd_f = self.predictor_func ( inp_f0, inp_f1, inp_f2 )

        out_img = np.clip(prd_f, 0.0, 1.0)

        out_img = np.concatenate ( [cv2.resize ( inp_f1, (prd_f.shape[1], prd_f.shape[0])  ), 
                                out_img], axis=1 )
        
        if debug:
            debugs += [out_img.copy()]
        
        return debugs if debug else out_img
