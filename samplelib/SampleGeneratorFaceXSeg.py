import multiprocessing
import pickle
import time
import traceback
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib, mplib, pathex
from core.imagelib import sd
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor, SampleType)

class SampleGeneratorFaceXSeg(SampleGeneratorBase):
    def __init__ (self, paths, debug=False, batch_size=1, resolution=256, face_type=None,
                        generators_count=4, data_format="NHWC",
                        **kwargs):

        super().__init__(debug, batch_size)
        self.initialized = False

        samples = []
        for path in paths:
            samples += SampleLoader.load (SampleType.FACE, path)

        seg_samples = [ sample for sample in samples if sample.seg_ie_polys.get_pts_count() != 0]
        seg_samples_len = len(seg_samples)
        if seg_samples_len == 0:
            raise Exception(f"No segmented faces found.")
        else:
            io.log_info(f"Using {seg_samples_len} segmented samples.")
            
        pickled_samples = pickle.dumps(seg_samples, 4)
        
        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (pickled_samples, resolution, face_type, data_format) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (pickled_samples, resolution, face_type, data_format), start_now=False ) \
                               for i in range(self.generators_count) ]

            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1

        self.initialized = True

    #overridable
    def is_initialized(self):
        return self.initialized

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        pickled_samples, resolution, face_type, data_format = param

        samples = pickle.loads(pickled_samples)    
            
        shuffle_idxs = []
        idxs = [*range(len(samples))]
        
        random_flip = True
        rotation_range=[-10,10]
        scale_range=[-0.05, 0.05]
        tx_range=[-0.05, 0.05]
        ty_range=[-0.05, 0.05]

        random_bilinear_resize_chance, random_bilinear_resize_max_size_per = 25,75
        motion_blur_chance, motion_blur_mb_max_size = 25, 5
        gaussian_blur_chance, gaussian_blur_kernel_max_size = 25, 5

        bs = self.batch_size
        while True:
            batches = [ [], [] ]

            n_batch = 0
            while n_batch < bs:
                try:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = idxs.copy()
                        np.random.shuffle(shuffle_idxs)
                    idx = shuffle_idxs.pop()
                    
                    sample = samples[idx] 

                    img = sample.load_bgr()
                    h,w,c = img.shape

                    mask = np.zeros ((h,w,1), dtype=np.float32)
                    sample.seg_ie_polys.overlay_mask(mask)

                    warp_params = imagelib.gen_warp_params(resolution, random_flip, rotation_range=rotation_range, scale_range=scale_range, tx_range=tx_range, ty_range=ty_range )

                    if face_type == sample.face_type:
                        if w != resolution:
                            img = cv2.resize( img, (resolution, resolution), cv2.INTER_LANCZOS4 )
                            mask = cv2.resize( mask, (resolution, resolution), cv2.INTER_LANCZOS4 )
                    else:
                        mat = LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, face_type)
                        img  = cv2.warpAffine( img,  mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )
                        mask = cv2.warpAffine( mask, mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LANCZOS4 )

                    if len(mask.shape) == 2:
                        mask = mask[...,None]

                    img   = imagelib.warp_by_params (warp_params, img,  can_warp=True, can_transform=True, can_flip=True, border_replicate=False)
                    mask  = imagelib.warp_by_params (warp_params, mask, can_warp=True, can_transform=True, can_flip=True, border_replicate=False)

                    img = np.clip(img.astype(np.float32), 0, 1)
                    mask[mask < 0.5] = 0.0
                    mask[mask >= 0.5] = 1.0
                    mask = np.clip(mask, 0, 1)

                    if np.random.randint(2) == 0:
                        img = imagelib.apply_random_hsv_shift(img, mask=sd.random_circle_faded ([resolution,resolution]))
                    else:                    
                        img = imagelib.apply_random_rgb_levels(img, mask=sd.random_circle_faded ([resolution,resolution]))
                    
                    img = imagelib.apply_random_motion_blur( img, motion_blur_chance, motion_blur_mb_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_gaussian_blur( img, gaussian_blur_chance, gaussian_blur_kernel_max_size, mask=sd.random_circle_faded ([resolution,resolution]))
                    img = imagelib.apply_random_bilinear_resize( img, random_bilinear_resize_chance, random_bilinear_resize_max_size_per, mask=sd.random_circle_faded ([resolution,resolution]))
                    
                    if data_format == "NCHW":
                        img = np.transpose(img, (2,0,1) )
                        mask = np.transpose(mask, (2,0,1) )

                    batches[0].append ( img )
                    batches[1].append ( mask )

                    n_batch += 1
                except:
                    io.log_err ( traceback.format_exc() )

            yield [ np.array(batch) for batch in batches]
