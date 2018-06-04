from facelib import FaceType
from facelib import LandmarksProcessor
import cv2
import numpy as np
from models import TrainingDataGeneratorBase
from utils import image_utils
from utils import random_utils
from enum import IntEnum
from models import TrainingDataType

class TrainingDataGenerator(TrainingDataGeneratorBase):
    class SampleTypeFlags(IntEnum):
        SOURCE               = 0x000001,
        WARPED               = 0x000002,
        WARPED_TRANSFORMED   = 0x000004,
        TRANSFORMED          = 0x000008,
   
        HALF_FACE            = 0x000010,
        FULL_FACE            = 0x000020,
        HEAD_FACE            = 0x000040,
        AVATAR_FACE          = 0x000080,
        MARK_ONLY_FACE       = 0x000100,
        
        MODE_BGR             = 0x001000,  #BGR
        MODE_G               = 0x002000,  #Grayscale
        MODE_GGG             = 0x004000,  #3xGrayscale 
        MODE_M               = 0x008000,  #mask only
        MODE_BGR_SHUFFLE     = 0x010000,  #BGR shuffle

        MASK_FULL            = 0x100000,
        MASK_EYES            = 0x200000,
        
    #overrided
    def onInitialize(self, random_flip=False, normalize_tanh=False, rotation_range=[-10,10], scale_range=[-0.05, 0.05], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05], output_sample_types=[], **kwargs):
        self.random_flip = random_flip        
        self.normalize_tanh = normalize_tanh
        self.output_sample_types = output_sample_types
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.tx_range = tx_range
        self.ty_range = ty_range
        
    #overrided
    def onProcessSample(self, sample, debug):
        source = sample.load_bgr()
        h,w,c = source.shape
        
        is_face_sample = self.trainingdatatype >= TrainingDataType.FACE_BEGIN and self.trainingdatatype <= TrainingDataType.FACE_END

        if debug and is_face_sample:
            LandmarksProcessor.draw_landmarks (source, sample.landmarks, (0, 1, 0))

        params = image_utils.gen_warp_params(source, self.random_flip, rotation_range=self.rotation_range, scale_range=self.scale_range, tx_range=self.tx_range, ty_range=self.ty_range )

        images = [[None]*3 for _ in range(4)]
            
        outputs = []        
        for t,size in self.output_sample_types:
            if t & self.SampleTypeFlags.SOURCE != 0:
                img_type = 0
            elif t & self.SampleTypeFlags.WARPED != 0:
                img_type = 1
            elif t & self.SampleTypeFlags.WARPED_TRANSFORMED != 0:
                img_type = 2
            elif t & self.SampleTypeFlags.TRANSFORMED != 0:
                img_type = 3
            else:
                raise ValueError ('expected SampleTypeFlags type')
                
            mask_type = 0
            if t & self.SampleTypeFlags.MASK_FULL != 0:
                mask_type = 1               
            elif t & self.SampleTypeFlags.MASK_EYES != 0:
                mask_type = 2
                    
            if images[img_type][mask_type] is None:
                img = source
                if is_face_sample:
                    if mask_type == 1:
                        img = np.concatenate( (img, LandmarksProcessor.get_image_hull_mask (source, sample.landmarks) ), -1 )                    
                    elif mask_type == 2:
                        mask = LandmarksProcessor.get_image_eye_mask (source, sample.landmarks)
                        mask = np.expand_dims (cv2.blur (mask, ( w // 32, w // 32 ) ), -1)
                        mask[mask > 0.0] = 1.0
                        img = np.concatenate( (img, mask ), -1 )               

                images[img_type][mask_type] = image_utils.warp_by_params (params, img, (img_type==1 or img_type==2), (img_type==2 or img_type==3), img_type != 0)
                
            img = images[img_type][mask_type]

            target_face_type = -1
            if t & self.SampleTypeFlags.HALF_FACE != 0:
                target_face_type = FaceType.HALF            
            elif t & self.SampleTypeFlags.FULL_FACE != 0:
                target_face_type = FaceType.FULL
            elif t & self.SampleTypeFlags.HEAD_FACE != 0:
                target_face_type = FaceType.HEAD
            elif t & self.SampleTypeFlags.AVATAR_FACE != 0:
                target_face_type = FaceType.AVATAR
            elif t & self.SampleTypeFlags.MARK_ONLY_FACE != 0:
                target_face_type = FaceType.MARK_ONLY
                    
            if is_face_sample and target_face_type != -1 and target_face_type != FaceType.MARK_ONLY:
                if target_face_type > sample.face_type:
                    raise Exception ('sample %s type %s does not match model requirement %s. Consider extract necessary type of faces.' % (sample.filename, sample.face_type, target_face_type) )
                
                img = cv2.warpAffine( img, LandmarksProcessor.get_transform_mat (sample.landmarks, size, target_face_type), (size,size), flags=cv2.INTER_LANCZOS4 )
            else:
                img = cv2.resize( img, (size,size), cv2.INTER_LANCZOS4 )
                
            img_bgr  = img[...,0:3]
            img_mask = img[...,3:4]
 
            if t & self.SampleTypeFlags.MODE_BGR != 0:
                img = img
            elif t & self.SampleTypeFlags.MODE_BGR_SHUFFLE != 0:
                img_bgr = np.take (img_bgr, np.random.permutation(img_bgr.shape[-1]), axis=-1)
                img = np.concatenate ( (img_bgr,img_mask) , -1 )
            elif t & self.SampleTypeFlags.MODE_G != 0:
                img = np.concatenate ( (np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1),img_mask) , -1 )
            elif t & self.SampleTypeFlags.MODE_GGG != 0:
                img = np.concatenate ( ( np.repeat ( np.expand_dims(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),-1), (3,), -1), img_mask), -1)
            elif is_face_sample and t & self.SampleTypeFlags.MODE_M != 0:
                if mask_type== 0:
                    raise ValueError ('no mask mode defined')
                img = img_mask
            else:
                raise ValueError ('expected SampleTypeFlags mode')
     
            if not debug and self.normalize_tanh:
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
   
    
    