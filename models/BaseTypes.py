from enum import IntEnum
import cv2
import numpy as np
from random import randint
from facelib import FaceType


class TrainingDataType(IntEnum):
    IMAGE = 0 #raw image
    
    FACE_BEGIN = 1
    FACE = 1                      #aligned face unsorted
    FACE_YAW_SORTED = 2           #sorted by yaw
    FACE_YAW_SORTED_AS_TARGET = 3 #sorted by yaw and included only yaws which exist in TARGET also automatic mirrored
    FACE_END = 3
    
    QTY = 4
    
    
class TrainingDataSample(object):

    def __init__(self, filename=None, face_type=None, shape=None, landmarks=None, yaw=None, mirror=None, nearest_target_list=None):
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        self.yaw = yaw
        self.mirror = mirror
        self.nearest_target_list = nearest_target_list
    
    def copy_and_set(self, filename=None, face_type=None, shape=None, landmarks=None, yaw=None, mirror=None, nearest_target_list=None):
        return TrainingDataSample( 
            filename=filename if filename is not None else self.filename, 
            face_type=face_type if face_type is not None else self.face_type, 
            shape=shape if shape is not None else self.shape, 
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(), 
            yaw=yaw if yaw is not None else self.yaw, 
            mirror=mirror if mirror is not None else self.mirror, 
            nearest_target_list=nearest_target_list if nearest_target_list is not None else self.nearest_target_list)
    
    def load_bgr(self):
        img = cv2.imread (self.filename).astype(np.float32) / 255.0
        if self.mirror:
            img = img[:,::-1].copy()
        return img

    def get_random_nearest_target_sample(self):
        if self.nearest_target_list is None:
            return None
        return self.nearest_target_list[randint (0, len(self.nearest_target_list)-1)]