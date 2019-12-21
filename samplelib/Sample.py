from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG


class SampleType(IntEnum):
    IMAGE = 0 #raw image

    FACE_BEGIN = 1
    FACE = 1                        #aligned face unsorted
    FACE_TEMPORAL_SORTED = 2        #sorted by source filename
    FACE_END = 2

    QTY = 5

class Sample(object):
    __slots__ = ['sample_type',
                 'filename',
                 'person_id',
                 'face_type',
                 'shape',
                 'landmarks',
                 'ie_polys',
                 'pitch_yaw_roll',
                 'eyebrows_expand_mod',
                 'source_filename',
                 'mirror',
                 'fanseg_mask_exist',
                 '_filename_offset_size',
                ]
    
    def __init__(self, sample_type=None, filename=None, person_id=None, face_type=None, shape=None, landmarks=None, ie_polys=None, pitch_yaw_roll=None, eyebrows_expand_mod=None, source_filename=None, mirror=None, fanseg_mask_exist=False):
        self.sample_type = sample_type if sample_type is not None else SampleType.IMAGE
        self.filename = filename
        self.person_id = person_id
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        self.ie_polys = ie_polys
        self.pitch_yaw_roll = pitch_yaw_roll
        self.eyebrows_expand_mod = eyebrows_expand_mod
        self.source_filename = source_filename
        self.mirror = mirror
        self.fanseg_mask_exist = fanseg_mask_exist
        
        self._filename_offset_size = None
        
    def set_filename_offset_size(self, filename, offset, size):
        self._filename_offset_size = (filename, offset, size)

    def copy_and_set(self, sample_type=None, filename=None, person_id=None, face_type=None, shape=None, landmarks=None, ie_polys=None, pitch_yaw_roll=None, eyebrows_expand_mod=None, source_filename=None, mirror=None, fanseg_mask=None, fanseg_mask_exist=None):
        return Sample(
            sample_type=sample_type if sample_type is not None else self.sample_type,
            filename=filename if filename is not None else self.filename,
            person_id=person_id if person_id is not None else self.person_id,
            face_type=face_type if face_type is not None else self.face_type,
            shape=shape if shape is not None else self.shape,
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(),
            ie_polys=ie_polys if ie_polys is not None else self.ie_polys,
            pitch_yaw_roll=pitch_yaw_roll if pitch_yaw_roll is not None else self.pitch_yaw_roll,
            eyebrows_expand_mod=eyebrows_expand_mod if eyebrows_expand_mod is not None else self.eyebrows_expand_mod,
            source_filename=source_filename if source_filename is not None else self.source_filename,
            mirror=mirror if mirror is not None else self.mirror,
            fanseg_mask_exist=fanseg_mask_exist if fanseg_mask_exist is not None else self.fanseg_mask_exist)

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
        if self.mirror:
            img = img[:,::-1].copy()
        return img

    def load_fanseg_mask(self):
        if self.fanseg_mask_exist:
            filepath = Path(self.filename)
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load ( str(filepath), loader_func=self.read_raw_file )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath), loader_func=self.read_raw_file )
            else:
                dflimg = None
            return dflimg.get_fanseg_mask()

        return None

