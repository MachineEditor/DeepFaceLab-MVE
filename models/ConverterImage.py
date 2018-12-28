from models import ConverterBase
from facelib import LandmarksProcessor
from facelib import FaceType

import cv2
import numpy as np
from utils import image_utils

'''
predictor: 
    input:  [predictor_input_size, predictor_input_size, BGR]
    output: [predictor_input_size, predictor_input_size, BGR]
'''

class ConverterImage(ConverterBase):

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
    def convert_image (self, img_bgr, debug):
        img_size = img_bgr.shape[1], img_bgr.shape[0]

        predictor_input_bgr = cv2.resize ( img_bgr, (self.predictor_input_size, self.predictor_input_size), cv2.INTER_LANCZOS4 )
        predicted_bgr = self.predictor ( predictor_input_bgr )

        output = cv2.resize ( predicted_bgr, (self.output_size, self.output_size), cv2.INTER_LANCZOS4 )
        if debug:
            return (predictor_input_bgr,output,)
        return output
