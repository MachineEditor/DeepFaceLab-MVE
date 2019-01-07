from enum import IntEnum
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils import Path_utils
from utils.DFLPNG import DFLPNG

from .Sample import Sample
from .Sample import SampleType

from facelib import FaceType
from facelib import LandmarksProcessor

class SampleLoader:
    cache = dict()
    
    @staticmethod
    def load(sample_type, samples_path, target_samples_path=None):
        cache = SampleLoader.cache
        
        if str(samples_path) not in cache.keys():
            cache[str(samples_path)] = [None]*SampleType.QTY
            
        if target_samples_path is not None and str(target_samples_path) not in cache.keys():
            cache[str(target_samples_path)] = [None]*SampleType.QTY
            
        datas = cache[str(samples_path)]

        if            sample_type == SampleType.IMAGE:
            if  datas[sample_type] is None:  
                datas[sample_type] = [ Sample(filename=filename) for filename in tqdm( Path_utils.get_image_paths(samples_path), desc="Loading" ) ]

        elif          sample_type == SampleType.FACE:
            if  datas[sample_type] is None:  
                datas[sample_type] = SampleLoader.upgradeToFaceSamples( [ Sample(filename=filename) for filename in Path_utils.get_image_paths(samples_path) ] )
        
        elif          sample_type == SampleType.FACE_YAW_SORTED:
            if  datas[sample_type] is None:
                datas[sample_type] = SampleLoader.upgradeToFaceYawSortedSamples( SampleLoader.load(SampleType.FACE, samples_path) )
        
        elif          sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:            
            if  datas[sample_type] is None:
                if target_samples_path is None:
                    raise Exception('target_samples_path is None for FACE_YAW_SORTED_AS_TARGET')
                datas[sample_type] = SampleLoader.upgradeToFaceYawSortedAsTargetSamples( SampleLoader.load(SampleType.FACE_YAW_SORTED, samples_path), SampleLoader.load(SampleType.FACE_YAW_SORTED, target_samples_path) )
            
        return datas[sample_type]
        
    @staticmethod
    def upgradeToFaceSamples ( samples ):
        sample_list = []
        
        for s in tqdm( samples, desc="Loading" ):

            s_filename_path = Path(s.filename)
            if s_filename_path.suffix != '.png':
                print ("%s is not a png file required for training" % (s_filename_path.name) ) 
                continue
            
            dflpng = DFLPNG.load ( str(s_filename_path), print_on_no_embedded_data=True )
            if dflpng is None:
                continue

            sample_list.append( s.copy_and_set(sample_type=SampleType.FACE,
                                               face_type=FaceType.fromString (dflpng.get_face_type()),
                                               shape=dflpng.get_shape(), 
                                               landmarks=dflpng.get_landmarks(),
                                               yaw=dflpng.get_yaw_value()) )
            
        return sample_list
        
    @staticmethod
    def upgradeToFaceYawSortedSamples( samples ):

        lowest_yaw, highest_yaw = -256, +256      
        gradations = 64
        diff_rot_per_grad = abs(highest_yaw-lowest_yaw) / gradations

        yaws_sample_list = [None]*gradations
        
        for i in tqdm( range(0, gradations), desc="Sorting" ):
            yaw = lowest_yaw + i*diff_rot_per_grad
            next_yaw = lowest_yaw + (i+1)*diff_rot_per_grad

            yaw_samples = []        
            for s in samples:                
                s_yaw = s.yaw
                if (i == 0            and s_yaw < next_yaw) or \
                   (i  < gradations-1 and s_yaw >= yaw and s_yaw < next_yaw) or \
                   (i == gradations-1 and s_yaw >= yaw):
                    yaw_samples.append ( s.copy_and_set(sample_type=SampleType.FACE_YAW_SORTED) )
                    
            if len(yaw_samples) > 0:
                yaws_sample_list[i] = yaw_samples
        
        return yaws_sample_list
        
    @staticmethod
    def upgradeToFaceYawSortedAsTargetSamples (s, t):
        l = len(s)
        if l != len(t):
            raise Exception('upgradeToFaceYawSortedAsTargetSamples() s_len != t_len')
        b = l // 2
        
        s_idxs = np.argwhere ( np.array ( [ 1 if x != None else 0  for x in s] ) == 1 )[:,0]
        t_idxs = np.argwhere ( np.array ( [ 1 if x != None else 0  for x in t] ) == 1 )[:,0]
        
        new_s = [None]*l    
        
        for t_idx in t_idxs:
            search_idxs = []        
            for i in range(0,l):
                search_idxs += [t_idx - i, (l-t_idx-1) - i, t_idx + i, (l-t_idx-1) + i]

            for search_idx in search_idxs:            
                if search_idx in s_idxs:
                    mirrored = ( t_idx != search_idx and ((t_idx < b and search_idx >= b) or (search_idx < b and t_idx >= b)) )
                    new_s[t_idx] = [ sample.copy_and_set(sample_type=SampleType.FACE_YAW_SORTED_AS_TARGET,
                                                         mirror=True, 
                                                         yaw=-sample.yaw, 
                                                         landmarks=LandmarksProcessor.mirror_landmarks (sample.landmarks, sample.shape[1] ))
                                          for sample in s[search_idx] 
                                        ] if mirrored else s[search_idx]                
                    break
                 
        return new_s