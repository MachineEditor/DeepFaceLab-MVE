import collections
import math
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.imagelib import sd
from facelib import FaceType, LandmarksProcessor


class SampleProcessor(object):
    class SampleType(IntEnum):
        NONE = 0
        IMAGE = 1
        FACE_IMAGE = 2
        FACE_MASK  = 3
        LANDMARKS_ARRAY            = 4
        PITCH_YAW_ROLL             = 5
        PITCH_YAW_ROLL_SIGMOID     = 6

    class ChannelType(IntEnum):
        NONE = 0
        BGR                   = 1  #BGR
        G                     = 2  #Grayscale
        GGG                   = 3  #3xGrayscale

    class FaceMaskType(IntEnum):
        NONE          = 0
        FULL_FACE      = 1  #mask all hull as grayscale
        EYES     = 2  #mask eyes hull as grayscale
        FULL_FACE_EYES = 3  #combo all + eyes as grayscale

    class Options(object):
        def __init__(self, random_flip = True, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05] ):
            self.random_flip = random_flip
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range

    @staticmethod
    def process (samples, sample_process_options, output_sample_types, debug, ct_sample=None):
        SPST = SampleProcessor.SampleType
        SPCT = SampleProcessor.ChannelType
        SPFMT = SampleProcessor.FaceMaskType

        sample_rnd_seed = np.random.randint(0x80000000)

        outputs = []
        for sample in samples:
            sample_face_type = sample.face_type
            sample_bgr = sample.load_bgr()
            sample_landmarks = sample.landmarks
            ct_sample_bgr = None
            h,w,c = sample_bgr.shape
            
            def get_full_face_mask():   
                xseg_mask = sample.get_xseg_mask()                                     
                if xseg_mask is not None:           
                    if xseg_mask.shape[0] != h or xseg_mask.shape[1] != w:
                        xseg_mask = cv2.resize(xseg_mask, (w,h), interpolation=cv2.INTER_CUBIC)                    
                        xseg_mask = imagelib.normalize_channels(xseg_mask, 1)
                    return np.clip(xseg_mask, 0, 1)
                else:
                    full_face_mask = LandmarksProcessor.get_image_hull_mask (sample_bgr.shape, sample_landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                    return np.clip(full_face_mask, 0, 1)
                
            def get_eyes_mask():
                eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample_landmarks)
                return np.clip(eyes_mask, 0, 1)

            is_face_sample = sample_landmarks is not None

            if debug and is_face_sample:
                LandmarksProcessor.draw_landmarks (sample_bgr, sample_landmarks, (0, 1, 0))
        
            params_per_resolution = {}            
            warp_rnd_state = np.random.RandomState (sample_rnd_seed-1)            
            for opts in output_sample_types:
                resolution = opts.get('resolution', None)
                if resolution is None:
                    continue
                params_per_resolution[resolution] = imagelib.gen_warp_params(resolution, 
                                                                             sample_process_options.random_flip, 
                                                                             rotation_range=sample_process_options.rotation_range, 
                                                                             scale_range=sample_process_options.scale_range, 
                                                                             tx_range=sample_process_options.tx_range, 
                                                                             ty_range=sample_process_options.ty_range, 
                                                                             rnd_state=warp_rnd_state)

            outputs_sample = []
            for opts in output_sample_types:
                sample_type    = opts.get('sample_type', SPST.NONE)
                channel_type   = opts.get('channel_type', SPCT.NONE)                
                resolution     = opts.get('resolution', 0)
                nearest_resize_to = opts.get('nearest_resize_to', None)
                warp           = opts.get('warp', False)
                transform      = opts.get('transform', False)
                motion_blur    = opts.get('motion_blur', None)
                gaussian_blur  = opts.get('gaussian_blur', None)
                random_bilinear_resize = opts.get('random_bilinear_resize', None)
                random_rgb_levels = opts.get('random_rgb_levels', False)
                random_hsv_shift = opts.get('random_hsv_shift', False)
                random_circle_mask = opts.get('random_circle_mask', False)
                normalize_tanh = opts.get('normalize_tanh', False)
                ct_mode        = opts.get('ct_mode', None)
                data_format    = opts.get('data_format', 'NHWC')
                
                if sample_type == SPST.FACE_MASK or sample_type == SPST.IMAGE: 
                    border_replicate = False
                elif sample_type == SPST.FACE_IMAGE:
                    border_replicate = True
                    
                    
                border_replicate = opts.get('border_replicate', border_replicate)
                borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
                
                
                if sample_type == SPST.FACE_IMAGE or sample_type == SPST.FACE_MASK:
                    if not is_face_sample:    
                        raise ValueError("face_samples should be provided for sample_type FACE_*")

                if sample_type == SPST.FACE_IMAGE or sample_type == SPST.FACE_MASK:
                    face_type      = opts.get('face_type', None)
                    face_mask_type = opts.get('face_mask_type', SPFMT.NONE)
                
                    if face_type is None:
                        raise ValueError("face_type must be defined for face samples")

                    if sample_type == SPST.FACE_MASK: 
                        if face_mask_type == SPFMT.FULL_FACE:
                            img = get_full_face_mask()
                        elif face_mask_type == SPFMT.EYES:
                            img = get_eyes_mask()
                        elif face_mask_type == SPFMT.FULL_FACE_EYES:
                            img = get_full_face_mask()                            
                            img += get_eyes_mask()*img
                        else:
                            img = np.zeros ( sample_bgr.shape[0:2]+(1,), dtype=np.float32)

                        if sample_face_type == FaceType.MARK_ONLY:
                            mat  = LandmarksProcessor.get_transform_mat (sample_landmarks, warp_resolution, face_type)
                            img = cv2.warpAffine( img, mat, (warp_resolution, warp_resolution), flags=cv2.INTER_LINEAR )
                            
                            img = imagelib.warp_by_params (params_per_resolution[resolution], img, warp, transform, can_flip=True, border_replicate=border_replicate, cv2_inter=cv2.INTER_LINEAR)
                            img = cv2.resize( img, (resolution,resolution), interpolation=cv2.INTER_LINEAR )
                        else:
                            if face_type != sample_face_type:
                                mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, face_type)                            
                                img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_LINEAR )
                            else:
                                if w != resolution:
                                    img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_LINEAR )
                                
                            img = imagelib.warp_by_params (params_per_resolution[resolution], img, warp, transform, can_flip=True, border_replicate=border_replicate, cv2_inter=cv2.INTER_LINEAR)

                        if len(img.shape) == 2:
                            img = img[...,None]
                            
                        if channel_type == SPCT.G:
                            out_sample = img.astype(np.float32)
                        else:
                            raise ValueError("only channel_type.G supported for the mask")

                    elif sample_type == SPST.FACE_IMAGE:
                        img = sample_bgr                      
                        
                        if random_rgb_levels:
                            random_mask = sd.random_circle_faded ([w,w], rnd_state=np.random.RandomState (sample_rnd_seed) ) if random_circle_mask else None
                            img = imagelib.apply_random_rgb_levels(img, mask=random_mask, rnd_state=np.random.RandomState (sample_rnd_seed) )

                        if random_hsv_shift:
                            random_mask = sd.random_circle_faded ([w,w], rnd_state=np.random.RandomState (sample_rnd_seed+1) ) if random_circle_mask else None
                            img = imagelib.apply_random_hsv_shift(img, mask=random_mask, rnd_state=np.random.RandomState (sample_rnd_seed+1) )

                            
                        if face_type != sample_face_type:
                            mat = LandmarksProcessor.get_transform_mat (sample_landmarks, resolution, face_type)
                            img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=borderMode, flags=cv2.INTER_CUBIC )
                        else:
                            if w != resolution:
                                img = cv2.resize( img, (resolution, resolution), interpolation=cv2.INTER_CUBIC )
                                
                        # Apply random color transfer                        
                        if ct_mode is not None and ct_sample is not None:
                            if ct_sample_bgr is None:
                               ct_sample_bgr = ct_sample.load_bgr()
                            img = imagelib.color_transfer (ct_mode, img, cv2.resize( ct_sample_bgr, (resolution,resolution), interpolation=cv2.INTER_LINEAR ) )

                        
                        img  = imagelib.warp_by_params (params_per_resolution[resolution], img,  warp, transform, can_flip=True, border_replicate=border_replicate)
  
                        img = np.clip(img.astype(np.float32), 0, 1)
   

                        
                        
                        if motion_blur is not None:                            
                            random_mask = sd.random_circle_faded ([resolution,resolution], rnd_state=np.random.RandomState (sample_rnd_seed+2)) if random_circle_mask else None
                            img = imagelib.apply_random_motion_blur(img, *motion_blur, mask=random_mask,rnd_state=np.random.RandomState (sample_rnd_seed+2) )

                        if gaussian_blur is not None:
                            random_mask = sd.random_circle_faded ([resolution,resolution], rnd_state=np.random.RandomState (sample_rnd_seed+3)) if random_circle_mask else None
                            img = imagelib.apply_random_gaussian_blur(img, *gaussian_blur, mask=random_mask,rnd_state=np.random.RandomState (sample_rnd_seed+3) )
                                
                        if random_bilinear_resize is not None:
                            random_mask = sd.random_circle_faded ([resolution,resolution], rnd_state=np.random.RandomState (sample_rnd_seed+4)) if random_circle_mask else None
                            img = imagelib.apply_random_bilinear_resize(img, *random_bilinear_resize, mask=random_mask,rnd_state=np.random.RandomState (sample_rnd_seed+4) )
                            
                             
                            
                        # Transform from BGR to desired channel_type
                        if channel_type == SPCT.BGR:
                            out_sample = img
                        elif channel_type == SPCT.G:
                            out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                        elif channel_type == SPCT.GGG:
                            out_sample = np.repeat ( np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),-1), (3,), -1)

                    # Final transformations
                    if nearest_resize_to is not None:
                        out_sample = cv2_resize(out_sample, (nearest_resize_to,nearest_resize_to), interpolation=cv2.INTER_NEAREST)
                        
                    if not debug:
                        if normalize_tanh:
                            out_sample = np.clip (out_sample * 2.0 - 1.0, -1.0, 1.0)
                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )
                elif sample_type == SPST.IMAGE:
                    img = sample_bgr      
                    img  = imagelib.warp_by_params (params_per_resolution[resolution], img,  warp, transform, can_flip=True, border_replicate=True)
                    img  = cv2.resize( img,  (resolution, resolution), interpolation=cv2.INTER_CUBIC )
                    out_sample = img
                    
                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )
                    
                    
                elif sample_type == SPST.LANDMARKS_ARRAY:
                    l = sample_landmarks
                    l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
                    l = np.clip(l, 0.0, 1.0)
                    out_sample = l
                elif sample_type == SPST.PITCH_YAW_ROLL or sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                    pitch,yaw,roll = sample.get_pitch_yaw_roll()
                    if params_per_resolution[resolution]['flip']:
                        yaw = -yaw

                    if sample_type == SPST.PITCH_YAW_ROLL_SIGMOID:
                        pitch = np.clip( (pitch / math.pi) / 2.0 + 0.5, 0, 1)
                        yaw   = np.clip( (yaw / math.pi) / 2.0 + 0.5, 0, 1)
                        roll  = np.clip( (roll / math.pi) / 2.0 + 0.5, 0, 1)

                    out_sample = (pitch, yaw)
                else:
                    raise ValueError ('expected sample_type')

                outputs_sample.append ( out_sample )
            outputs += [outputs_sample]

        return outputs

"""

        STRUCT        = 4  #mask structure as grayscale
                           elif face_mask_type == SPFMT.STRUCT:
                            if sample.eyebrows_expand_mod is not None:
                                img = LandmarksProcessor.get_face_struct_mask (sample_bgr.shape, sample_landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                            else:
                                img = LandmarksProcessor.get_face_struct_mask (sample_bgr.shape, sample_landmarks)
     
        
        
        close_sample = sample.close_target_list[ np.random.randint(0, len(sample.close_target_list)) ] if sample.close_target_list is not None else None
        close_sample_bgr = close_sample.load_bgr() if close_sample is not None else None

        if debug and close_sample_bgr is not None:
            LandmarksProcessor.draw_landmarks (close_sample_bgr, close_sample.landmarks, (0, 1, 0))
        RANDOM_CLOSE               = 0x00000040, #currently unused
        MORPH_TO_RANDOM_CLOSE      = 0x00000080, #currently unused

if f & SPTF.RANDOM_CLOSE != 0:
                img_type += 10
            elif f & SPTF.MORPH_TO_RANDOM_CLOSE != 0:
                img_type += 20
if img_type >= 10 and img_type <= 19: #RANDOM_CLOSE
    img_type -= 10
    img = close_sample_bgr
    cur_sample = close_sample

elif img_type >= 20 and img_type <= 29: #MORPH_TO_RANDOM_CLOSE
    img_type -= 20
    res = sample.shape[0]

    s_landmarks = sample.landmarks.copy()
    d_landmarks = close_sample.landmarks.copy()
    idxs = list(range(len(s_landmarks)))
    #remove landmarks near boundaries
    for i in idxs[:]:
        s_l = s_landmarks[i]
        d_l = d_landmarks[i]
        if s_l[0] < 5 or s_l[1] < 5 or s_l[0] >= res-5 or s_l[1] >= res-5 or \
            d_l[0] < 5 or d_l[1] < 5 or d_l[0] >= res-5 or d_l[1] >= res-5:
            idxs.remove(i)
    #remove landmarks that close to each other in 5 dist
    for landmarks in [s_landmarks, d_landmarks]:
        for i in idxs[:]:
            s_l = landmarks[i]
            for j in idxs[:]:
                if i == j:
                    continue
                s_l_2 = landmarks[j]
                diff_l = np.abs(s_l - s_l_2)
                if np.sqrt(diff_l.dot(diff_l)) < 5:
                    idxs.remove(i)
                    break
    s_landmarks = s_landmarks[idxs]
    d_landmarks = d_landmarks[idxs]
    s_landmarks = np.concatenate ( [s_landmarks, [ [0,0], [ res // 2, 0], [ res-1, 0], [0, res//2], [res-1, res//2] ,[0,res-1] ,[res//2, res-1] ,[res-1,res-1] ] ] )
    d_landmarks = np.concatenate ( [d_landmarks, [ [0,0], [ res // 2, 0], [ res-1, 0], [0, res//2], [res-1, res//2] ,[0,res-1] ,[res//2, res-1] ,[res-1,res-1] ] ] )
    img = imagelib.morph_by_points (sample_bgr, s_landmarks, d_landmarks)
    cur_sample = close_sample
else:
    """
