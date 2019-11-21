import multiprocessing
import shutil
from pathlib import Path

import cv2
import numpy as np

from facelib import FaceType, LandmarksProcessor
from interact import interact as io
from joblib import Subprocessor
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG

from . import Extractor, Sorter
from .Extractor import ExtractSubprocessor


def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')
    
    bb_csv = input_path / 'loose_bb_train.csv'
    if not bb_csv.exists():
        raise ValueError('loose_bb_train.csv found. Please ensure it exists.')
    
    bb_lines = bb_csv.read_text().split('\n')
    bb_lines.pop(0)
    
    bb_dict = {}
    for line in bb_lines:
        name, l, t, w, h = line.split(',')
        name = name[1:-1]
        l, t, w, h = [ int(x) for x in (l, t, w, h) ]        
        bb_dict[name] = (l,t,w, h)
    

    output_path = input_path.parent / (input_path.name + '_out')
    
    dir_names = Path_utils.get_all_dir_names(input_path)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    data = []
    for dir_name in io.progress_bar_generator(dir_names, "Collecting"):
        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name
        
        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)
            
        input_path_image_paths = Path_utils.get_image_paths(cur_input_path)

        for filename in input_path_image_paths:
            filename_path = Path(filename)
            
            name = filename_path.parent.name + '/' + filename_path.stem
            if name not in bb_dict:
                continue

            l,t,w,h = bb_dict[name]
            if min(w,h) < 128:
                continue
            
            data += [ ExtractSubprocessor.Data(filename=filename,rects=[ (l,t,l+w,t+h) ], landmarks_accurate=False, force_output_path=cur_output_path ) ]
            
    face_type = FaceType.fromString('full_face')
    
    io.log_info ('Performing 2nd pass...')
    data = ExtractSubprocessor (data, 'landmarks', 256, face_type, debug_dir=None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).run()
       
    io.log_info ('Performing 3rd pass...')
    ExtractSubprocessor (data, 'final', 256, face_type, debug_dir=None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=None).run()
    
    
"""
    import code
    code.interact(local=dict(globals(), **locals()))
    
    data_len = len(data) 
    i = 0
    while i < data_len-1:
        i_name = Path(data[i].filename).parent.name
        
        sub_data = []
        
        for j in range (i, data_len):
            j_name = Path(data[j].filename).parent.name
            if i_name == j_name:
                sub_data += [ data[j] ]
            else:
                break
        i = j
        
        cur_output_path = output_path / i_name        
        
        io.log_info (f"Processing: {str(cur_output_path)}, {i}/{data_len} ")
        
        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)
            
            
        



    
    
    for dir_name in dir_names:
        
        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name
        
        input_path_image_paths = Path_utils.get_image_paths(cur_input_path)
        l = len(input_path_image_paths)
        #if l < 250 or l > 350:
        #    continue

        io.log_info (f"Processing: {str(cur_input_path)} ")
        
        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)


        data = []
        for filename in input_path_image_paths:
            filename_path = Path(filename)
            
            name = filename_path.parent.name + '/' + filename_path.stem
            if name not in bb_dict:
                continue
            
            bb = bb_dict[name]
            l,t,w,h = bb
            if min(w,h) < 128:
                continue
            
            data += [ ExtractSubprocessor.Data(filename=filename,rects=[ (l,t,l+w,t+h) ], landmarks_accurate=False ) ]
                
    
              
        io.log_info ('Performing 2nd pass...')
        data = ExtractSubprocessor (data, 'landmarks', 256, face_type, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False).run()

        io.log_info ('Performing 3rd pass...')
        data = ExtractSubprocessor (data, 'final', 256, face_type, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False, final_output_path=cur_output_path).run()
        
        
        io.log_info (f"Sorting: {str(cur_output_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')
        
        import code
        code.interact(local=dict(globals(), **locals()))
        
        #try:
        #    io.log_info (f"Removing: {str(cur_input_path)} ")
        #    shutil.rmtree(cur_input_path)
        #except:
        #    io.log_info (f"unable to remove: {str(cur_input_path)} ")
            
            
            

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
            
"""            

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
