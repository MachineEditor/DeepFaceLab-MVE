from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from utils.cv2_utils import *
from DFLIMG import *
from facelib import LandmarksProcessor
from imagelib import IEPolys

class SampleType(IntEnum):
    IMAGE = 0 #raw image

    FACE_BEGIN = 1
    FACE = 1                        #aligned face unsorted
    FACE_PERSON = 2                 #aligned face person
    FACE_TEMPORAL_SORTED = 3        #sorted by source filename
    FACE_END = 3

    QTY = 4

class Sample(object):
    __slots__ = ['sample_type',
                 'filename',
                 'face_type',
                 'shape',
                 'landmarks',
                 'ie_polys',
                 'eyebrows_expand_mod',
                 'source_filename',
                 'person_name',
                 'pitch_yaw_roll',
                 '_filename_offset_size',                 
                ]

    def __init__(self, sample_type=None,
                       filename=None,
                       face_type=None,
                       shape=None,
                       landmarks=None,
                       ie_polys=None,                       
                       eyebrows_expand_mod=None,
                       source_filename=None,
                       person_name=None,                       
                       pitch_yaw_roll=None,
                       **kwargs):

        self.sample_type = sample_type if sample_type is not None else SampleType.IMAGE
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        self.ie_polys = IEPolys.load(ie_polys)
        self.eyebrows_expand_mod = eyebrows_expand_mod
        self.source_filename = source_filename
        self.person_name = person_name
        self.pitch_yaw_roll = pitch_yaw_roll 
        
        self._filename_offset_size = None
 
    def get_pitch_yaw_roll(self):
        if self.pitch_yaw_roll is None:
            self.pitch_yaw_roll = LandmarksProcessor.estimate_pitch_yaw_roll(landmarks)
        return self.pitch_yaw_roll
        
    def set_filename_offset_size(self, filename, offset, size):
        self._filename_offset_size = (filename, offset, size)

    def read_raw_file(self, filename=None):
        if self._filename_offset_size is not None:
            filename, offset, size = self._filename_offset_size
            with open(filename, "rb") as f:
                f.seek( offset, 0)
                return f.read (size)
        else:
            with open(filename, "rb") as f:
                return f.read()

    def load_bgr(self):
        img = cv2_imread (self.filename, loader_func=self.read_raw_file).astype(np.float32) / 255.0
        return img

    def get_config(self):
        return {'sample_type': self.sample_type,
                'filename': self.filename,
                'face_type': self.face_type,
                'shape': self.shape,
                'landmarks': self.landmarks.tolist(),
                'ie_polys': self.ie_polys.dump(),
                'eyebrows_expand_mod': self.eyebrows_expand_mod,
                'source_filename': self.source_filename,
                'person_name': self.person_name
               }

"""
def copy_and_set(self, sample_type=None, filename=None, face_type=None, shape=None, landmarks=None, ie_polys=None, pitch_yaw_roll=None, eyebrows_expand_mod=None, source_filename=None, fanseg_mask=None, person_name=None):
        return Sample(
            sample_type=sample_type if sample_type is not None else self.sample_type,
            filename=filename if filename is not None else self.filename,
            face_type=face_type if face_type is not None else self.face_type,
            shape=shape if shape is not None else self.shape,
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(),
            ie_polys=ie_polys if ie_polys is not None else self.ie_polys,
            pitch_yaw_roll=pitch_yaw_roll if pitch_yaw_roll is not None else self.pitch_yaw_roll,
            eyebrows_expand_mod=eyebrows_expand_mod if eyebrows_expand_mod is not None else self.eyebrows_expand_mod,
            source_filename=source_filename if source_filename is not None else self.source_filename,
            person_name=person_name if person_name is not None else self.person_name)

"""