import collections
from enum import IntEnum

import cv2
import numpy as np

from core import imagelib
from facelib import FaceType, LandmarksProcessor

class SampleProcessor(object):
    class Types(IntEnum):
        NONE = 0

        IMG_TYPE_BEGIN = 1
        IMG_SOURCE                     = 1
        IMG_WARPED                     = 2
        IMG_WARPED_TRANSFORMED         = 3
        IMG_TRANSFORMED                = 4
        IMG_LANDMARKS_ARRAY            = 5 #currently unused
        IMG_PITCH_YAW_ROLL             = 6
        IMG_PITCH_YAW_ROLL_SIGMOID     = 7
        IMG_TYPE_END = 10

        FACE_TYPE_BEGIN = 10
        FACE_TYPE_HALF             = 10
        FACE_TYPE_MID_FULL         = 11
        FACE_TYPE_FULL             = 12
        FACE_TYPE_WHOLE_FACE       = 13
        FACE_TYPE_HEAD             = 14  #currently unused
        FACE_TYPE_AVATAR           = 15  #currently unused
        FACE_TYPE_FULL_NO_ALIGN    = 16
        FACE_TYPE_HEAD_NO_ALIGN    = 17
        FACE_TYPE_END = 20

        MODE_BEGIN = 40
        MODE_BGR                   = 40  #BGR
        MODE_G                     = 41  #Grayscale
        MODE_GGG                   = 42  #3xGrayscale
        MODE_FACE_MASK_ALL_HULL    = 43  #mask all hull as grayscale
        MODE_FACE_MASK_EYES_HULL   = 44  #mask eyes hull as grayscale
        MODE_FACE_MASK_ALL_EYES_HULL = 45  #combo all + eyes as grayscale
        MODE_FACE_MASK_STRUCT      = 46  #mask structure as grayscale
        MODE_BGR_SHUFFLE           = 47  #BGR shuffle
        MODE_BGR_RANDOM_HSV_SHIFT  = 48
        MODE_BGR_RANDOM_RGB_LEVELS = 49
        MODE_END = 50

    class Options(object):
        def __init__(self, random_flip = True, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05] ):
            self.random_flip = random_flip
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range

    SPTF_FACETYPE_TO_FACETYPE =  {  Types.FACE_TYPE_HALF : FaceType.HALF,
                                    Types.FACE_TYPE_MID_FULL : FaceType.MID_FULL,
                                    Types.FACE_TYPE_FULL : FaceType.FULL,
                                    Types.FACE_TYPE_WHOLE_FACE : FaceType.WHOLE_FACE,
                                    Types.FACE_TYPE_HEAD : FaceType.HEAD,
                                    Types.FACE_TYPE_FULL_NO_ALIGN : FaceType.FULL_NO_ALIGN,
                                    Types.FACE_TYPE_HEAD_NO_ALIGN : FaceType.HEAD_NO_ALIGN,
                                 }

    @staticmethod
    def process (samples, sample_process_options, output_sample_types, debug, ct_sample=None):
        SPTF = SampleProcessor.Types

        sample_rnd_seed = np.random.randint(0x80000000)
        
        outputs = []
        for sample in samples:
            sample_bgr = sample.load_bgr()
            ct_sample_bgr = None
            h,w,c = sample_bgr.shape

            is_face_sample = sample.landmarks is not None

            if debug and is_face_sample:
                LandmarksProcessor.draw_landmarks (sample_bgr, sample.landmarks, (0, 1, 0))

            params = imagelib.gen_warp_params(sample_bgr, sample_process_options.random_flip, rotation_range=sample_process_options.rotation_range, scale_range=sample_process_options.scale_range, tx_range=sample_process_options.tx_range, ty_range=sample_process_options.ty_range )

            outputs_sample = []
            for opts in output_sample_types:

                resolution = opts.get('resolution', 0)
                types = opts.get('types', [] )

                motion_blur = opts.get('motion_blur', None)
                gaussian_blur = opts.get('gaussian_blur', None)

                ct_mode = opts.get('ct_mode', 'None')
                normalize_tanh = opts.get('normalize_tanh', False)
                data_format = opts.get('data_format', 'NHWC')


                img_type = SPTF.NONE
                target_face_type = SPTF.NONE
                mode_type = SPTF.NONE
                for t in types:
                    if t >= SPTF.IMG_TYPE_BEGIN and t < SPTF.IMG_TYPE_END:
                        img_type = t
                    elif t >= SPTF.FACE_TYPE_BEGIN and t < SPTF.FACE_TYPE_END:
                        target_face_type = t
                    elif t >= SPTF.MODE_BEGIN and t < SPTF.MODE_END:
                        mode_type = t


                if is_face_sample:
                    if target_face_type == SPTF.NONE:
                         raise ValueError("target face type must be defined for face samples")
                else:
                    if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL:
                        raise ValueError("MODE_FACE_MASK_ALL_HULL applicable only for face samples")
                    if mode_type == SPTF.MODE_FACE_MASK_EYES_HULL:
                        raise ValueError("MODE_FACE_MASK_EYES_HULL applicable only for face samples")
                    if mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL:
                        raise ValueError("MODE_FACE_MASK_ALL_EYES_HULL applicable only for face samples")
                    if mode_type == SPTF.MODE_FACE_MASK_STRUCT:
                        raise ValueError("MODE_FACE_MASK_STRUCT applicable only for face samples")

                can_warp      = (img_type==SPTF.IMG_WARPED or img_type==SPTF.IMG_WARPED_TRANSFORMED)
                can_transform = (img_type==SPTF.IMG_WARPED_TRANSFORMED or img_type==SPTF.IMG_TRANSFORMED)

                if img_type == SPTF.NONE:
                    raise ValueError ('expected IMG_ type')

                if img_type == SPTF.IMG_LANDMARKS_ARRAY:
                    l = sample.landmarks
                    l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
                    l = np.clip(l, 0.0, 1.0)
                    out_sample = l
                elif img_type == SPTF.IMG_PITCH_YAW_ROLL or img_type == SPTF.IMG_PITCH_YAW_ROLL_SIGMOID:
                    pitch_yaw_roll = sample.get_pitch_yaw_roll()

                    if params['flip']:
                        yaw = -yaw

                    if img_type == SPTF.IMG_PITCH_YAW_ROLL_SIGMOID:
                        pitch = np.clip( (pitch / math.pi) / 2.0 + 0.5, 0, 1)
                        yaw   = np.clip( (yaw / math.pi) / 2.0 + 0.5, 0, 1)
                        roll  = np.clip( (roll / math.pi) / 2.0 + 0.5, 0, 1)

                    out_sample = (pitch, yaw, roll)
                else:
                    if mode_type == SPTF.NONE:
                        raise ValueError ('expected MODE_ type')

                    if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL or \
                       mode_type == SPTF.MODE_FACE_MASK_EYES_HULL or \
                       mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL:

                        if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL or \
                           mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL:
                            if sample.eyebrows_expand_mod is not None:
                                all_mask = LandmarksProcessor.get_image_hull_mask (sample_bgr.shape, sample.landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                            else:
                                all_mask = LandmarksProcessor.get_image_hull_mask (sample_bgr.shape, sample.landmarks)
                            
                            all_mask = np.clip(all_mask, 0, 1)
                            
                        if mode_type == SPTF.MODE_FACE_MASK_EYES_HULL or \
                           mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL:
                            eyes_mask = LandmarksProcessor.get_image_eye_mask (sample_bgr.shape, sample.landmarks)
                            eyes_mask = np.clip(eyes_mask, 0, 1)
                            
                        if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL:
                            img = all_mask
                        elif mode_type == SPTF.MODE_FACE_MASK_EYES_HULL:
                            img = eyes_mask
                        elif mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL:
                            img = all_mask + eyes_mask
                            
                        if sample.ie_polys is not None:
                            sample.ie_polys.overlay_mask(img)

                    elif mode_type == SPTF.MODE_FACE_MASK_STRUCT:
                        if sample.eyebrows_expand_mod is not None:
                            img = LandmarksProcessor.get_face_struct_mask (sample_bgr.shape, sample.landmarks, eyebrows_expand_mod=sample.eyebrows_expand_mod )
                        else:
                            img = LandmarksProcessor.get_face_struct_mask (sample_bgr.shape, sample.landmarks)
                    else:
                        img = sample_bgr
                        if motion_blur is not None:
                            chance, mb_max_size = motion_blur
                            chance = np.clip(chance, 0, 100)

                            l_rnd_state = np.random.RandomState (sample_rnd_seed)
                            mblur_rnd_chance = l_rnd_state.randint(100)
                            mblur_rnd_kernel = l_rnd_state.randint(mb_max_size)+1
                            mblur_rnd_deg    = l_rnd_state.randint(360)

                            if mblur_rnd_chance < chance:
                                img = imagelib.LinearMotionBlur (img, mblur_rnd_kernel, mblur_rnd_deg )

                        if gaussian_blur is not None:
                            chance, kernel_max_size = gaussian_blur
                            chance = np.clip(chance, 0, 100)
                            
                            l_rnd_state = np.random.RandomState (sample_rnd_seed+1)
                            gblur_rnd_chance = l_rnd_state.randint(100)
                            gblur_rnd_kernel = l_rnd_state.randint(kernel_max_size)*2+1

                            if gblur_rnd_chance < chance:
                                img = cv2.GaussianBlur(img, (gblur_rnd_kernel,) *2 , 0)

                    if is_face_sample:
                        target_ft = SampleProcessor.SPTF_FACETYPE_TO_FACETYPE[target_face_type]
                        if target_ft > sample.face_type:
                            raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, target_ft) )

                        if sample.face_type == FaceType.MARK_ONLY:
                            mat  = LandmarksProcessor.get_transform_mat (sample.landmarks, sample.shape[0], target_ft)

                            if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL or \
                               mode_type == SPTF.MODE_FACE_MASK_EYES_HULL or \
                               mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL or \
                               mode_type == SPTF.MODE_FACE_MASK_STRUCT:
                                img = cv2.warpAffine( img, mat, (sample.shape[0],sample.shape[0]), flags=cv2.INTER_LINEAR )
                                img = imagelib.warp_by_params (params, img, can_warp, can_transform, can_flip=True, border_replicate=False, cv2_inter=cv2.INTER_LINEAR)
                                img = cv2.resize( img, (resolution,resolution), cv2.INTER_LINEAR )[...,None]
                            else:
                                img  = cv2.warpAffine( img,  mat, (sample.shape[0],sample.shape[0]), flags=cv2.INTER_CUBIC )
                                img  = imagelib.warp_by_params (params, img,  can_warp, can_transform, can_flip=True, border_replicate=True)
                                img  = cv2.resize( img,  (resolution,resolution), cv2.INTER_CUBIC )

                        else:
                            mat = LandmarksProcessor.get_transform_mat (sample.landmarks, resolution, target_ft)

                            if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL or \
                               mode_type == SPTF.MODE_FACE_MASK_EYES_HULL or \
                               mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL or \
                               mode_type == SPTF.MODE_FACE_MASK_STRUCT:                                
                                img = imagelib.warp_by_params (params, img, can_warp, can_transform, can_flip=True, border_replicate=False, cv2_inter=cv2.INTER_LINEAR)
                                img = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_LINEAR )[...,None]                                
                            else:
                                img  = imagelib.warp_by_params (params, img,  can_warp, can_transform, can_flip=True, border_replicate=True)
                                img  = cv2.warpAffine( img, mat, (resolution,resolution), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC )
                    else:
                        img  = imagelib.warp_by_params (params, img,  can_warp, can_transform, can_flip=True, border_replicate=True)
                        img  = cv2.resize( img,  (resolution,resolution), cv2.INTER_CUBIC )


                    if mode_type == SPTF.MODE_FACE_MASK_ALL_HULL or \
                       mode_type == SPTF.MODE_FACE_MASK_EYES_HULL or \
                       mode_type == SPTF.MODE_FACE_MASK_ALL_EYES_HULL or \
                       mode_type == SPTF.MODE_FACE_MASK_STRUCT:
                        out_sample = img.astype(np.float32)
                    else:
                        img = np.clip(img.astype(np.float32), 0, 1)

                        if ct_mode is not None and ct_sample is not None:
                            if ct_sample_bgr is None:
                                ct_sample_bgr = ct_sample.load_bgr()
                            img = imagelib.color_transfer (ct_mode,
                                                           img,
                                                           cv2.resize( ct_sample_bgr, (resolution,resolution), cv2.INTER_LINEAR ) )

                        if mode_type == SPTF.MODE_BGR:
                            out_sample = img
                        elif mode_type == SPTF.MODE_BGR_SHUFFLE:
                            l_rnd_state = np.random.RandomState (sample_rnd_seed)
                            out_sample = np.take (img, l_rnd_state.permutation(img.shape[-1]), axis=-1)

                        elif mode_type == SPTF.MODE_BGR_RANDOM_HSV_SHIFT:
                            l_rnd_state = np.random.RandomState (sample_rnd_seed)
                            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv)
                            h = (h + l_rnd_state.randint(360) ) % 360
                            s = np.clip ( s + l_rnd_state.random()-0.5, 0, 1 )
                            v = np.clip ( v + l_rnd_state.random()-0.5, 0, 1 )
                            hsv = cv2.merge([h, s, v])
                            out_sample = np.clip( cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) , 0, 1 )
                            
                        elif mode_type == SPTF.MODE_BGR_RANDOM_RGB_LEVELS:
                            l_rnd_state = np.random.RandomState (sample_rnd_seed)
                            np_rnd = l_rnd_state.rand                            
                            
                            inBlack  = np.array([np_rnd()*0.25    , np_rnd()*0.25    , np_rnd()*0.25], dtype=np.float32)
                            inWhite  = np.array([1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25, 1.0-np_rnd()*0.25], dtype=np.float32)
                            inGamma  = np.array([0.5+np_rnd(), 0.5+np_rnd(), 0.5+np_rnd()], dtype=np.float32)
                            outBlack = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            outWhite = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                            
                            out_sample = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 1 )                            
                            out_sample = ( out_sample ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
                            out_sample = np.clip(out_sample, 0, 1)
                        elif mode_type == SPTF.MODE_G:
                            out_sample = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[...,None]
                        elif mode_type == SPTF.MODE_GGG:
                            out_sample = np.repeat ( np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),-1), (3,), -1)

                    if not debug:
                        if normalize_tanh:
                            out_sample = np.clip (out_sample * 2.0 - 1.0, -1.0, 1.0)

                    if data_format == "NCHW":
                        out_sample = np.transpose(out_sample, (2,0,1) )

                outputs_sample.append ( out_sample )
            outputs += [outputs_sample]

        return outputs

"""
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
