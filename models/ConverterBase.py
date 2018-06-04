import copy
'''
You can implement your own Converter, check example ConverterMasked.py
'''

class ConverterBase(object):
    MODE_FACE = 0
    MODE_IMAGE = 1
    
    #overridable
    def __init__(self, predictor):
        self.predictor = predictor
        
    #overridable
    def get_mode(self):
        #MODE_FACE calls convert_face
        #MODE_IMAGE calls convert_image
        return ConverterBase.MODE_FACE
        
    #overridable
    def convert_face (self, img_bgr, img_face_landmarks, debug):
        #return float32 image        
        #if debug , return tuple ( images of any size and channels, ...)
        return image
        
    #overridable
    def convert_image (self, img_bgr, img_landmarks, debug):
        #img_landmarks not None, if input image is png with embedded data
        #return float32 image        
        #if debug , return tuple ( images of any size and channels, ...)
        return image
        
    #overridable
    def dummy_predict(self):
        #do dummy predict here
        pass

    def copy(self):
        return copy.copy(self)
        
    def copy_and_set_predictor(self, predictor):
        result = self.copy()
        result.predictor = predictor
        return result