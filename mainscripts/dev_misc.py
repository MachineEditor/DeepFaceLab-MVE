import multiprocessing
import shutil
from pathlib import Path

import cv2
import numpy as np

from facelib import LandmarksProcessor
from interact import interact as io
from joblib import Subprocessor
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG

from . import Extractor, Sorter


def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_out')
    
    dir_names = Path_utils.get_all_dir_names(input_path)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        
    
    
    for dir_name in dir_names:
        
        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name
        
        l = len(Path_utils.get_image_paths(cur_input_path))
        if l < 250 or l > 350:
            continue

        io.log_info (f"Processing: {str(cur_input_path)} ")
        
        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)

        Extractor.main( str(cur_input_path),
              str(cur_output_path),
              detector='s3fd',
              image_size=256,
              face_type='full_face',
              max_faces_from_image=1,
              device_args=device_args )
              
        io.log_info (f"Sorting: {str(cur_input_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')
        
        try:
            io.log_info (f"Removing: {str(cur_input_path)} ")
            shutil.rmtree(cur_input_path)
        except:
            io.log_info (f"unable to remove: {str(cur_input_path)} ")
            
            

class CelebAMASKHQSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.masks_files_paths = client_dict['masks_files_paths']
            return None

        #override
        def process_data(self, data):
            filename = data[0]
            filepath = Path(filename)
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load( str(filepath) )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath) )
            else:
                dflimg = None
                
            image_to_face_mat = dflimg.get_image_to_face_mat()    
            src_filename = dflimg.get_source_filename()

            img = cv2_imread(filename)
            h,w,c = img.shape
            
            fanseg_mask = LandmarksProcessor.get_image_hull_mask(img.shape, dflimg.get_landmarks() )
            
            idx_name = '%.5d' % int(src_filename.split('.')[0])
            idx_files = [ x for x in self.masks_files_paths if idx_name in x ]  
                
            skin_files = [ x for x in idx_files if 'skin' in x ]
            eye_glass_files = [ x for x in idx_files if 'eye_g' in x ]
            
            for files, is_invert in [ (skin_files,False), 
                                      (eye_glass_files,True) ]:
                if len(files) > 0:
                    mask = cv2_imread(files[0])            
                    mask = mask[...,0]
                    mask[mask == 255] = 1
                    mask = mask.astype(np.float32)
                    mask = cv2.resize(mask, (1024,1024) )                
                    mask = cv2.warpAffine(mask, image_to_face_mat, (w, h), cv2.INTER_LANCZOS4)
                    
                    if not is_invert:
                        fanseg_mask *= mask[...,None]
                    else:
                        fanseg_mask *= (1-mask[...,None])

            dflimg.embed_and_set (filename, fanseg_mask=fanseg_mask)
            return 1

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, image_paths, masks_files_paths ):
        self.image_paths = image_paths
        self.masks_files_paths = masks_files_paths
        
        self.result = []
        super().__init__('CelebAMASKHQSubprocessor', CelebAMASKHQSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % (i), {}, {'masks_files_paths' : self.masks_files_paths }

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Processing", len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.image_paths) > 0:
            return [self.image_paths.pop(0)]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.image_paths.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result
        
#unused in end user workflow
def apply_celebamaskhq(input_dir ):
    
    input_path = Path(input_dir)    
    
    img_path = input_path / 'aligned'
    mask_path = input_path / 'mask'

    if not img_path.exists():
        raise ValueError(f'{str(img_path)} directory not found. Please ensure it exists.')

    CelebAMASKHQSubprocessor(Path_utils.get_image_paths(img_path), 
                             Path_utils.get_image_paths(mask_path, subdirs=True) ).run()
    
    return
    
    paths_to_extract = []
    for filename in io.progress_bar_generator(Path_utils.get_image_paths(img_path), desc="Processing"):
        filepath = Path(filename)
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is not None:
            paths_to_extract.append (filepath)
            
        image_to_face_mat = dflimg.get_image_to_face_mat()    
        src_filename = dflimg.get_source_filename()

        #img = cv2_imread(filename)
        h,w,c = dflimg.get_shape()
        
        fanseg_mask = LandmarksProcessor.get_image_hull_mask( (h,w,c), dflimg.get_landmarks() )
        
        idx_name = '%.5d' % int(src_filename.split('.')[0])
        idx_files = [ x for x in masks_files if idx_name in x ]  
              
        skin_files = [ x for x in idx_files if 'skin' in x ]
        eye_glass_files = [ x for x in idx_files if 'eye_g' in x ]
        
        for files, is_invert in [ (skin_files,False), 
                                  (eye_glass_files,True) ]:
                       
            if len(files) > 0:
                mask = cv2_imread(files[0])            
                mask = mask[...,0]
                mask[mask == 255] = 1
                mask = mask.astype(np.float32)
                mask = cv2.resize(mask, (1024,1024) )                
                mask = cv2.warpAffine(mask, image_to_face_mat, (w, h), cv2.INTER_LANCZOS4)
                
                if not is_invert:
                    fanseg_mask *= mask[...,None]
                else:
                    fanseg_mask *= (1-mask[...,None])
            
        #cv2.imshow("", (fanseg_mask*255).astype(np.uint8) )
        #cv2.waitKey(0)    
        
        
        dflimg.embed_and_set (filename, fanseg_mask=fanseg_mask)
        
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
