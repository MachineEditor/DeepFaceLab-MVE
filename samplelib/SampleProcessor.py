from enum import IntEnum
import numpy as np
import cv2
import imagelib

from facelib import LandmarksProcessor
from facelib import FaceType

class SampleProcessor(object):
    class TypeFlags(IntEnum):
        SOURCE                     = 0x00000001,
        WARPED                     = 0x00000002,
        WARPED_TRANSFORMED         = 0x00000004,
        TRANSFORMED                = 0x00000008,
        LANDMARKS_ARRAY            = 0x00000010, #currently unused
        PITCH_YAW_ROLL             = 0x00000020,

        RANDOM_CLOSE               = 0x00000040, #currently unused
        MORPH_TO_RANDOM_CLOSE      = 0x00000080, #currently unused

        FACE_TYPE_HALF             = 0x00000100,
        FACE_TYPE_FULL             = 0x00000200,
        FACE_TYPE_HEAD             = 0x00000400,  #currently unused
        FACE_TYPE_AVATAR           = 0x00000800,  #currently unused

        FACE_MASK_FULL             = 0x00001000,
        FACE_MASK_EYES             = 0x00002000, #currently unused

        MODE_BGR                   = 0x00010000,  #BGR
        MODE_G                     = 0x00020000,  #Grayscale
        MODE_GGG                   = 0x00040000,  #3xGrayscale
        MODE_M                     = 0x00080000,  #mask only
        MODE_BGR_SHUFFLE           = 0x00100000,  #BGR shuffle

        OPT_APPLY_MOTION_BLUR      = 0x10000000,

    class Options(object):
        #motion_blur = [chance_int, range] - chance 0..100 to apply to face (not mask), and range [1..3] where 3 is highest power of motion blur

        def __init__(self, random_flip = True, normalize_tanh = False, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05], motion_blur=None ):
            self.random_flip = random_flip
            self.normalize_tanh = normalize_tanh
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range
            self.motion_blur = motion_blur
            if self.motion_blur is not None:
                chance, range = self.motion_blur
                chance = np.clip(chance, 0, 100)
                range = [3,5,7,9][ : np.clip(range, 0, 3)+1 ]
                self.motion_blur = (chance, range)

    @staticmethod
    def process (sample, sample_process_options, output_sample_types, debug):
        SPTF = SampleProcessor.TypeFlags

        sample_bgr = sample.load_bgr()
        h,w,c = sample_bgr.shape

        is_face_sample = sample.landmarks is not None

        if debug and is_face_sample:
            LandmarksProcessor.draw_landmarks (sample_bgr, sample.landmarks, (0, 1, 0))

        close_sample = sample.close_target_list[ np.random.randint(0, len(sample.close_target_list)) ] if sample.close_target_list is not None else None
        close_sample_bgr = close_sample.load_bgr() if close_sample is not None else None

        if debug and close_sample_bgr is not None:
            LandmarksProcessor.draw_landmarks (close_sample_bgr, close_sample.landmarks, (0, 1, 0))

        params = imagelib.gen_warp_params(sample_bgr, sample_process_options.random_flip, rotation_range=sample_process_options.rotation_range, scale_range=sample_process_options.scale_range, tx_range=sample_process_options.tx_range, ty_range=sample_process_options.ty_range )

        images = [[None]*3 for _ in range(30)]

        sample_rnd_seed = np.random.randint(0x80000000)

        outputs = []
        for sample_type in output_sample_types:
            f = sample_type[0]
            size = 0 if len (sample_type) < 2 else sample_type[1]
            random_sub_size = 0 if len (sample_type) < 3 else min( sample_type[2] , size)

            if f & SPTF.SOURCE != 0:
                img_type = 0
            elif f & SPTF.WARPED != 0:
                img_type = 1
            elif f & SPTF.WARPED_TRANSFORMED != 0:
                img_type = 2
            elif f & SPTF.TRANSFORMED != 0:
                img_type = 3
            elif f & SPTF.LANDMARKS_ARRAY != 0:
                img_type = 4
            elif f & SPTF.PITCH_YAW_ROLL != 0:
                img_type = 5
            else:
                raise ValueError ('expected SampleTypeFlags type')

            if f & SPTF.RANDOM_CLOSE != 0:
                img_type += 10
            elif f & SPTF.MORPH_TO_RANDOM_CLOSE != 0:
                img_type += 20

            face_mask_type = 0
            if f & SPTF.FACE_MASK_FULL != 0:
                face_mask_type = 1
            elif f & SPTF.FACE_MASK_EYES != 0:
                face_mask_type = 2

            target_face_type = -1
            if f & SPTF.FACE_TYPE_HALF != 0:
                target_face_type = FaceType.HALF
            elif f & SPTF.FACE_TYPE_FULL != 0:
                target_face_type = FaceType.FULL
            elif f & SPTF.FACE_TYPE_HEAD != 0:
                target_face_type = FaceType.HEAD
            elif f & SPTF.FACE_TYPE_AVATAR != 0:
                target_face_type = FaceType.AVATAR

            apply_motion_blur = f & SPTF.OPT_APPLY_MOTION_BLUR != 0

            if img_type == 4:
                l = sample.landmarks
                l = np.concatenate ( [ np.expand_dims(l[:,0] / w,-1), np.expand_dims(l[:,1] / h,-1) ], -1 )
                l = np.clip(l, 0.0, 1.0)
                img = l
            elif img_type == 5:
                pitch_yaw_roll = sample.pitch_yaw_roll
                if pitch_yaw_roll is not None:
                    pitch, yaw, roll = pitch_yaw_roll
                else:
                    pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll (sample.landmarks)
                if params['flip']:
                    yaw = -yaw
                    
                img = (pitch, yaw, roll)                
            else:
                if images[img_type][face_mask_type] is None:
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
                        img = sample_bgr
                        cur_sample = sample

                    if is_face_sample:
                        if apply_motion_blur and sample_process_options.motion_blur is not None:
                            chance, mb_range = sample_process_options.motion_blur
                            if np.random.randint(100) < chance :
                                dim = mb_range[ np.random.randint(len(mb_range) ) ]
                                img = imagelib.LinearMotionBlur (img, dim, np.random.randint(180) )

                        if face_mask_type == 1:
                            mask = cur_sample.load_fanseg_mask() #using fanseg_mask if exist

                            if mask is None:
                                mask = LandmarksProcessor.get_image_hull_mask (img.shape, cur_sample.landmarks)

                            if cur_sample.ie_polys is not None:
                                cur_sample.ie_polys.overlay_mask(mask)

                            img = np.concatenate( (img, mask ), -1 )
                        elif face_mask_type == 2:
                            mask = LandmarksProcessor.get_image_eye_mask (img.shape, cur_sample.landmarks)
                            mask = np.expand_dims (cv2.blur (mask, ( w // 32, w // 32 ) ), -1)
                            mask[mask > 0.0] = 1.0
                            img = np.concatenate( (img, mask ), -1 )

                    images[img_type][face_mask_type] = imagelib.warp_by_params (params, img, (img_type==1 or img_type==2), (img_type==2 or img_type==3), img_type != 0, face_mask_type == 0)

                img = images[img_type][face_mask_type]

                if is_face_sample and target_face_type != -1:
                    if target_face_type > sample.face_type:
                        raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, target_face_type) )
                    img = cv2.warpAffine( img, LandmarksProcessor.get_transform_mat (sample.landmarks, size, target_face_type), (size,size), flags=cv2.INTER_CUBIC )
                else:
                    img = cv2.resize( img, (size,size), cv2.INTER_CUBIC )

                if random_sub_size != 0:
                    sub_size = size - random_sub_size
                    rnd_state = np.random.RandomState (sample_rnd_seed+random_sub_size)
                    start_x = rnd_state.randint(sub_size+1)
                    start_y = rnd_state.randint(sub_size+1)
                    img = img[start_y:start_y+sub_size,start_x:start_x+sub_size,:]

                img_bgr  = img[...,0:3]
                img_mask = img[...,3:4]

                if f & SPTF.MODE_BGR != 0:
                    img = img_bgr
                elif f & SPTF.MODE_BGR_SHUFFLE != 0:
                    rnd_state = np.random.RandomState (sample_rnd_seed)
                    img_bgr = np.take (img_bgr, rnd_state.permutation(img_bgr.shape[-1]), axis=-1)
                    img = np.concatenate ( (img_bgr,img_mask) , -1 )
                elif f & SPTF.MODE_G != 0:
                    img = np.concatenate ( (np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1),img_mask) , -1 )
                elif f & SPTF.MODE_GGG != 0:
                    img = np.concatenate ( ( np.repeat ( np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1), (3,), -1), img_mask), -1)
                elif is_face_sample and f & SPTF.MODE_M != 0:
                    if face_mask_type== 0:
                        raise ValueError ('no face_mask_type defined')
                    img = img_mask
                else:
                    raise ValueError ('expected SampleTypeFlags mode')

                if not debug:
                    if sample_process_options.normalize_tanh:
                        img = np.clip (img * 2.0 - 1.0, -1.0, 1.0)
                    else:
                        img = np.clip (img, 0.0, 1.0)

            outputs.append ( img )

        if debug:
            result = []

            for output in outputs:
                if output.shape[2] < 4:
                    result += [output,]
                elif output.shape[2] == 4:
                    result += [output[...,0:3]*output[...,3:4],]

            return result
        else:
            return outputs
