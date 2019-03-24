import copy
'''
You can implement your own Converter, check example ConverterMasked.py
'''

class Converter(object):
    TYPE_FACE = 0                   #calls convert_face
    TYPE_IMAGE = 1                  #calls convert_image without landmarks
    TYPE_IMAGE_WITH_LANDMARKS = 2   #calls convert_image with landmarks

    #overridable
    def __init__(self, predictor_func, type):
        self.predictor_func = predictor_func
        self.type = type

    #overridable
    def on_cli_initialize(self):
        #cli initialization
        pass

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

    def copy_and_set_predictor(self, predictor_func):
        result = self.copy()
        result.predictor_func = predictor_func
        return result
