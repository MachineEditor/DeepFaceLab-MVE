from enum import IntEnum
import numpy as np
import cv2
from utils import image_utils
from facelib import LandmarksProcessor
from facelib import FaceType


class SampleProcessor(object):
    class TypeFlags(IntEnum):
        SOURCE               = 0x00000001,
        WARPED               = 0x00000002,
        WARPED_TRANSFORMED   = 0x00000004,
        TRANSFORMED          = 0x00000008,
   
        FACE_ALIGN_HALF      = 0x00000010,
        FACE_ALIGN_FULL      = 0x00000020,
        FACE_ALIGN_HEAD      = 0x00000040,
        FACE_ALIGN_AVATAR    = 0x00000080,        
        FACE_MASK_FULL       = 0x00000100,
        FACE_MASK_EYES       = 0x00000200,
        
        MODE_BGR             = 0x01000000,  #BGR
        MODE_G               = 0x02000000,  #Grayscale
        MODE_GGG             = 0x04000000,  #3xGrayscale 
        MODE_M               = 0x08000000,  #mask only
        MODE_BGR_SHUFFLE     = 0x10000000,  #BGR shuffle
   
    class Options(object):     
        def __init__(self, random_flip = True, normalize_tanh = False, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05]):
            self.random_flip = random_flip        
            self.normalize_tanh = normalize_tanh
            self.rotation_range = rotation_range
            self.scale_range = scale_range
            self.tx_range = tx_range
            self.ty_range = ty_range   
        
    @staticmethod
    def process (sample, sample_process_options, output_sample_types, debug):
        source = sample.load_bgr()
        h,w,c = source.shape
        
        is_face_sample = sample.landmarks is not None 
        
        if debug and is_face_sample:
            LandmarksProcessor.draw_landmarks (source, sample.landmarks, (0, 1, 0))

        params = image_utils.gen_warp_params(source, sample_process_options.random_flip, rotation_range=sample_process_options.rotation_range, scale_range=sample_process_options.scale_range, tx_range=sample_process_options.tx_range, ty_range=sample_process_options.ty_range )

        images = [[None]*3 for _ in range(4)]
            
        sample_rnd_seed = np.random.randint(0x80000000)
            
        outputs = []        
        for sample_type in output_sample_types:
            f = sample_type[0]
            size = sample_type[1]
            random_sub_size = 0 if len (sample_type) < 3 else min( sample_type[2] , size)
            
            if f & SampleProcessor.TypeFlags.SOURCE != 0:
                img_type = 0
            elif f & SampleProcessor.TypeFlags.WARPED != 0:
                img_type = 1
            elif f & SampleProcessor.TypeFlags.WARPED_TRANSFORMED != 0:
                img_type = 2
            elif f & SampleProcessor.TypeFlags.TRANSFORMED != 0:
                img_type = 3
            else:
                raise ValueError ('expected SampleTypeFlags type')
                
            face_mask_type = 0
            if f & SampleProcessor.TypeFlags.FACE_MASK_FULL != 0:
                face_mask_type = 1               
            elif f & SampleProcessor.TypeFlags.FACE_MASK_EYES != 0:
                face_mask_type = 2
                  
            target_face_type = -1
            if f & SampleProcessor.TypeFlags.FACE_ALIGN_HALF != 0:
                target_face_type = FaceType.HALF            
            elif f & SampleProcessor.TypeFlags.FACE_ALIGN_FULL != 0:
                target_face_type = FaceType.FULL
            elif f & SampleProcessor.TypeFlags.FACE_ALIGN_HEAD != 0:
                target_face_type = FaceType.HEAD
            elif f & SampleProcessor.TypeFlags.FACE_ALIGN_AVATAR != 0:
                target_face_type = FaceType.AVATAR
                
            if images[img_type][face_mask_type] is None:
                img = source
                if is_face_sample:
                    if face_mask_type == 1:
                        img = np.concatenate( (img, LandmarksProcessor.get_image_hull_mask (source, sample.landmarks) ), -1 )                    
                    elif face_mask_type == 2:
                        mask = LandmarksProcessor.get_image_eye_mask (source, sample.landmarks)
                        mask = np.expand_dims (cv2.blur (mask, ( w // 32, w // 32 ) ), -1)
                        mask[mask > 0.0] = 1.0
                        img = np.concatenate( (img, mask ), -1 )               

                images[img_type][face_mask_type] = image_utils.warp_by_params (params, img, (img_type==1 or img_type==2), (img_type==2 or img_type==3), img_type != 0)
                
            img = images[img_type][face_mask_type]
                    
            if is_face_sample and target_face_type != -1:
                if target_face_type > sample.face_type:
                    raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, target_face_type) )
                
                img = cv2.warpAffine( img, LandmarksProcessor.get_transform_mat (sample.landmarks, size, target_face_type), (size,size), flags=cv2.INTER_LANCZOS4 )
            else:
                img = cv2.resize( img, (size,size), cv2.INTER_LANCZOS4 )
                
            if random_sub_size != 0:
                sub_size = size - random_sub_size                
                rnd_state = np.random.RandomState (sample_rnd_seed+random_sub_size)
                start_x = rnd_state.randint(sub_size+1)
                start_y = rnd_state.randint(sub_size+1)
                img = img[start_y:start_y+sub_size,start_x:start_x+sub_size,:]

            img_bgr  = img[...,0:3]
            img_mask = img[...,3:4]

            if f & SampleProcessor.TypeFlags.MODE_BGR != 0:
                img = img
            elif f & SampleProcessor.TypeFlags.MODE_BGR_SHUFFLE != 0:
                img_bgr = np.take (img_bgr, np.random.permutation(img_bgr.shape[-1]), axis=-1)
                img = np.concatenate ( (img_bgr,img_mask) , -1 )
            elif f & SampleProcessor.TypeFlags.MODE_G != 0:
                img = np.concatenate ( (np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1),img_mask) , -1 )
            elif f & SampleProcessor.TypeFlags.MODE_GGG != 0:
                img = np.concatenate ( ( np.repeat ( np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1), (3,), -1), img_mask), -1)
            elif is_face_sample and f & SampleProcessor.TypeFlags.MODE_M != 0:
                if face_mask_type== 0:
                    raise ValueError ('no face_mask_type defined')
                img = img_mask
            else:
                raise ValueError ('expected SampleTypeFlags mode')
     
            if not debug and sample_process_options.normalize_tanh:
                img = img * 2.0 - 1.0
                
            outputs.append ( img )

        if debug:
            result = ()

            for output in outputs:
                if output.shape[2] < 4:
                    result += (output,)
                elif output.shape[2] == 4:
                    result += (output[...,0:3]*output[...,3:4],)

            return result            
        else:
            return outputs