import traceback
import os
import sys
import time
import multiprocessing
import shutil
from pathlib import Path
import numpy as np
import mathlib
import cv2
from utils import Path_utils
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from utils import image_utils
import facelib
from facelib import FaceType
from facelib import LandmarksProcessor
from nnlib import nnlib
from joblib import Subprocessor
from interact import interact as io
        
class ExtractSubprocessor(Subprocessor):
        
    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.log_info ('Running on %s.' % (client_dict['device_name']) )

            self.type         = client_dict['type']
            self.image_size   = client_dict['image_size']
            self.face_type    = client_dict['face_type']
            self.device_idx   = client_dict['device_idx']
            self.cpu_only     = client_dict['device_type'] == 'CPU'
            self.output_path  = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None        
            self.debug        = client_dict['debug']
            self.detector     = client_dict['detector']

            self.cached_image = (None, None)
            
            self.e = None
            device_config = nnlib.DeviceConfig ( cpu_only=self.cpu_only, force_gpu_idx=self.device_idx, allow_growth=True)
            if self.type == 'rects':
                if self.detector is not None:
                    if self.detector == 'mt':
                        nnlib.import_all (device_config)
                        self.e = facelib.MTCExtractor()                            
                    elif self.detector == 'dlib':
                        nnlib.import_dlib (device_config)
                        self.e = facelib.DLIBExtractor(nnlib.dlib)
                    elif self.detector == 's3fd':
                        nnlib.import_all (device_config)
                        self.e = facelib.S3FDExtractor()
                    else:
                        raise ValueError ("Wrond detector type.")
                        
                    if self.e is not None:
                        self.e.__enter__()
                    
            elif self.type == 'landmarks':
                nnlib.import_all (device_config)
                self.e = facelib.LandmarksExtractor(nnlib.keras)
                self.e.__enter__()
                
            elif self.type == 'final':
                pass
                
        #override
        def on_finalize(self):
            if self.e is not None:
                self.e.__exit__()
                
        #override
        def process_data(self, data):
            filename_path = Path( data[0] )

            filename_path_str = str(filename_path)
            if self.cached_image[0] == filename_path_str:
                image = self.cached_image[1]
            else:
                image = cv2_imread( filename_path_str )
                h, w, ch = image.shape
                wm = w % 2
                hm = h % 2
                if wm + hm != 0: #fix odd image
                    image = image[0:h-hm,0:w-wm,:]
                self.cached_image = ( filename_path_str, image )
            
            if image is None:
                self.log_err ( 'Failed to extract %s, reason: cv2_imread() fail.' % ( str(filename_path) ) )
            else:
                if self.type == 'rects':
                    h, w, ch = image.shape
                    if min(w,h) < 128:
                        self.log_err ( 'Image is too small %s : [%d, %d]' % ( str(filename_path), w, h ) )
                        rects = []
                    else:                    
                        rects = self.e.extract_from_bgr (image)
                        
                    return [str(filename_path), rects]

                elif self.type == 'landmarks':
                    rects = data[1]   
                    landmarks = self.e.extract_from_bgr (image, rects)                    
                    return [str(filename_path), landmarks]

                elif self.type == 'final':
                    src_dflimg = None
                    (h,w,c) = image.shape                
                    if h == w:
                        #extracting from already extracted jpg image?
                        if filename_path.suffix == '.jpg':
                            src_dflimg = DFLJPG.load ( str(filename_path) )
                
                    result = []
                    faces = data[1]
                    
                    if self.debug:
                        debug_output_file = '{}{}'.format( str(Path(str(self.output_path) + '_debug') / filename_path.stem),  '.jpg')
                        debug_image = image.copy()
                        
                    face_idx = 0
                    for face in faces:   
                        rect = np.array(face[0])
                        image_landmarks = np.array(face[1])

                        if self.face_type == FaceType.MARK_ONLY:                        
                            face_image = image
                            face_image_landmarks = image_landmarks
                        else:
                            image_to_face_mat = LandmarksProcessor.get_transform_mat (image_landmarks, self.image_size, self.face_type)       
                            face_image = cv2.warpAffine(image, image_to_face_mat, (self.image_size, self.image_size), cv2.INTER_LANCZOS4)
                            face_image_landmarks = LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)
                            
                            landmarks_bbox = LandmarksProcessor.transform_points ( [ (0,0), (0,self.image_size-1), (self.image_size-1, self.image_size-1), (self.image_size-1,0) ], image_to_face_mat, True)
                            
                            rect_area      = mathlib.polygon_area(np.array(rect[[0,2,2,0]]), np.array(rect[[1,1,3,3]]))
                            landmarks_area = mathlib.polygon_area(landmarks_bbox[:,0], landmarks_bbox[:,1] )
                            
                            if landmarks_area > 4*rect_area: #get rid of faces which umeyama-landmark-area > 4*detector-rect-area
                                continue

                        if self.debug:
                            LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, self.image_size, self.face_type)
                            
                        output_file = '{}_{}{}'.format(str(self.output_path / filename_path.stem), str(face_idx), '.jpg')
                        face_idx += 1
                        
                        if src_dflimg is not None:
                            #if extracting from dflimg just copy it in order not to lose quality
                            shutil.copy ( str(filename_path), str(output_file) )
                        else:
                            cv2_imwrite(output_file, face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85] )

                        DFLJPG.embed_data(output_file, face_type = FaceType.toString(self.face_type),
                                                       landmarks = face_image_landmarks.tolist(),
                                                       source_filename = filename_path.name,
                                                       source_rect=  rect,
                                                       source_landmarks = image_landmarks.tolist()
                                            )  
                            
                        result.append (output_file)
                        
                    if self.debug:
                        cv2_imwrite(debug_output_file, debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )
                        
                    return result       
            return None
             
        #overridable
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]
            
    #override
    def __init__(self, input_data, type, image_size, face_type, debug, multi_gpu=False, cpu_only=False, manual=False, manual_window_size=0, detector=None, output_path=None ): 
        self.input_data = input_data
        self.type = type
        self.image_size = image_size
        self.face_type = face_type
        self.debug = debug        
        self.multi_gpu = multi_gpu
        self.cpu_only = cpu_only
        self.detector = detector
        self.output_path = output_path        
        self.manual = manual        
        self.manual_window_size = manual_window_size
        self.result = []

        no_response_time_sec = 60 if not self.manual else 999999
        super().__init__('Extractor', ExtractSubprocessor.Cli, no_response_time_sec)

    #override
    def on_clients_initialized(self):
        if self.manual == True:
            self.wnd_name = 'Manual pass'
            io.named_window(self.wnd_name)
            io.capture_mouse(self.wnd_name)
            io.capture_keys(self.wnd_name)
            
            self.cache_original_image = (None, None)
            self.cache_image = (None, None)
            self.cache_text_lines_img = (None, None)
            self.hide_help = False
            
            self.landmarks = None
            self.x = 0
            self.y = 0
            self.rect_size = 100
            self.rect_locked = False
            self.redraw_needed = True
            
        io.progress_bar (None, len (self.input_data))
            
    #override
    def on_clients_finalized(self):
        if self.manual == True:
            io.destroy_all_windows()
            
        io.progress_bar_close()
        
    def get_devices_for_type (self, type, multi_gpu, cpu_only):
        if 'cpu' in nnlib.device.backend:
            cpu_only = True
            
        if not cpu_only and (type == 'rects' or type == 'landmarks'):
            if type == 'rects' and (self.detector == 'mt') and nnlib.device.backend == "plaidML":
                cpu_only = True
            else:
                if multi_gpu:
                    devices = nnlib.device.getValidDevicesWithAtLeastTotalMemoryGB(2)
                if not multi_gpu or len(devices) == 0:
                    devices = [nnlib.device.getBestValidDeviceIdx()]
                if len(devices) == 0:
                    devices = [0]
                    
                for idx in devices:
                    dev_name = nnlib.device.getDeviceName(idx)
                    dev_vram = nnlib.device.getDeviceVRAMTotalGb(idx)
                    
                    if not self.manual and ( self.type == 'rects' and self.detector != 's3fd' ):
                        for i in range ( int (max (1, dev_vram / 2) ) ):
                            yield (idx, 'GPU', '%s #%d' % (dev_name,i) , dev_vram)
                    else:
                        yield (idx, 'GPU', dev_name, dev_vram)

        if cpu_only and (type == 'rects' or type == 'landmarks'):
            if self.manual:
                yield (0, 'CPU', 'CPU', 0 )
            else:
                for i in range( min(8, multiprocessing.cpu_count() // 2) ):
                    yield (i, 'CPU', 'CPU%d' % (i), 0 )
                
        if type == 'final':
            for i in range( min(8, multiprocessing.cpu_count()) ):
                yield (i, 'CPU', 'CPU%d' % (i), 0 ) 
        
    #override
    def process_info_generator(self):
        base_dict = {'type' : self.type, 
                     'image_size': self.image_size, 
                     'face_type': self.face_type, 
                     'debug': self.debug, 
                     'output_dir': str(self.output_path), 
                     'detector': self.detector}
    
        for (device_idx, device_type, device_name, device_total_vram_gb) in self.get_devices_for_type(self.type, self.multi_gpu, self.cpu_only): 
            client_dict = base_dict.copy()
            client_dict['device_idx'] = device_idx
            client_dict['device_name'] = device_name
            client_dict['device_type'] = device_type            
            yield client_dict['device_name'], {}, client_dict

    #override
    def get_data(self, host_dict):
        if not self.manual:
            if len (self.input_data) > 0:
                return self.input_data.pop(0)    
        else:
            skip_remaining = False
            allow_remark_faces = False
            while len (self.input_data) > 0:
                data = self.input_data[0]
                filename, faces = data
                is_frame_done = False
                go_to_prev_frame = False

                # Can we mark an image that already has a marked face?
                if allow_remark_faces:
                    allow_remark_faces = False
                    # If there was already a face then lock the rectangle to it until the mouse is clicked
                    if len(faces) > 0:
                        self.rect, self.landmarks = faces.pop()
                        
                        self.rect_locked = True
                        self.redraw_needed = True
                        faces.clear()
                        self.rect_size = ( self.rect[2] - self.rect[0] ) / 2
                        self.x = ( self.rect[0] + self.rect[2] ) / 2
                        self.y = ( self.rect[1] + self.rect[3] ) / 2

                if len(faces) == 0:
                    if self.cache_original_image[0] == filename:
                        self.original_image = self.cache_original_image[1]
                    else:
                        self.original_image = cv2_imread( filename )
                        self.cache_original_image = (filename, self.original_image )
                    
                    (h,w,c) = self.original_image.shape
                    self.view_scale = 1.0 if self.manual_window_size == 0 else self.manual_window_size / ( h * (16.0/9.0) )
                    
                    if self.cache_image[0] == (h,w,c) + (self.view_scale,filename):
                        self.image = self.cache_image[1]
                    else:                                            
                        self.image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)    
                        self.cache_image = ( (h,w,c) + (self.view_scale,filename), self.image )
 
                    (h,w,c) = self.image.shape
                    
                    sh = (0,0, w, min(100, h) )                    
                    if self.cache_text_lines_img[0] == sh:
                        self.text_lines_img = self.cache_text_lines_img[1]
                    else:
                        self.text_lines_img = (image_utils.get_draw_text_lines ( self.image, sh,
                                                        [   'Match landmarks with face exactly. Click to confirm/unconfirm selection',
                                                            '[Enter] - confirm face landmarks and continue',
                                                            '[Space] - confirm as unmarked frame and continue',
                                                            '[Mouse wheel] - change rect',
                                                            '[,] [.]- prev frame, next frame. [Q] - skip remaining frames',
                                                            '[h] - hide this help'
                                                        ], (1, 1, 1) )*255).astype(np.uint8)
                                                        
                        self.cache_text_lines_img = (sh, self.text_lines_img)

                    while True:
                        new_x = self.x
                        new_y = self.y
                        new_rect_size = self.rect_size
                        
                        mouse_events = io.get_mouse_events(self.wnd_name)
                        for ev in mouse_events:
                            (x, y, ev, flags) = ev
                            if ev == io.EVENT_MOUSEWHEEL and not self.rect_locked:
                                mod = 1 if flags > 0 else -1 
                                diff = 1 if new_rect_size <= 40 else np.clip(new_rect_size / 10, 1, 10)
                                new_rect_size = max (5, new_rect_size + diff*mod)                    
                            elif ev == io.EVENT_LBUTTONDOWN:
                                self.rect_locked = not self.rect_locked
                                self.redraw_needed = True
                            elif not self.rect_locked:
                                new_x = np.clip (x, 0, w-1) / self.view_scale
                                new_y = np.clip (y, 0, h-1) / self.view_scale
                                
                        key_events = io.get_key_events(self.wnd_name)
                        key, = key_events[-1] if len(key_events) > 0 else (0,)

                        if key == ord('\r') or key == ord('\n'):
                            faces.append ( [(self.rect), self.landmarks] )
                            is_frame_done = True
                            break
                        elif key == ord(' '):
                            is_frame_done = True
                            break
                        elif key == ord('.'):
                            allow_remark_faces = True
                            # Only save the face if the rect is still locked
                            if self.rect_locked:
                                faces.append ( [(self.rect), self.landmarks] )
                            is_frame_done = True
                            break
                        elif key == ord(',')  and len(self.result) > 0:
                            # Only save the face if the rect is still locked
                            if self.rect_locked:
                                faces.append ( [(self.rect), self.landmarks] )
                            go_to_prev_frame = True
                            break
                        elif key == ord('q'):
                            skip_remaining = True
                            break
                        elif key == ord('h'):
                            self.hide_help = not self.hide_help
                            break
                        
                        if self.x != new_x or \
                           self.y != new_y or \
                           self.rect_size != new_rect_size or \
                           self.redraw_needed:
                            self.x = new_x
                            self.y = new_y
                            self.rect_size = new_rect_size

                            self.rect = ( int(self.x-self.rect_size), 
                                          int(self.y-self.rect_size), 
                                          int(self.x+self.rect_size), 
                                          int(self.y+self.rect_size) )
                                          
                            return [filename, [self.rect]]
                        
                        io.process_messages(0.0001)
                else:
                    is_frame_done = True
                    
                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    io.progress_bar_inc(1)
                    self.redraw_needed = True
                    self.rect_locked = False
                elif go_to_prev_frame:
                    self.input_data.insert(0, self.result.pop() )
                    io.progress_bar_inc(-1)
                    allow_remark_faces = True
                    self.redraw_needed = True
                    self.rect_locked = False
                elif skip_remaining:
                    if self.rect_locked:
                        faces.append ( [(self.rect), self.landmarks] )
                    while len(self.input_data) > 0:
                        self.result.append( self.input_data.pop(0) )
                        io.progress_bar_inc(1)

        return None
    
    #override
    def on_data_return (self, host_dict, data):
        if not self.manual:
            self.input_data.insert(0, data)   

    #override
    def on_result (self, host_dict, data, result):
        if self.manual == True:
            self.landmarks = result[1][0][1]
                                        
            (h,w,c) = self.image.shape
            
            if not self.hide_help:
                image = cv2.addWeighted (self.image,1.0,self.text_lines_img,1.0,0)
            else:
                image = self.image.copy()
                
            view_rect = (np.array(self.rect) * self.view_scale).astype(np.int).tolist()
            view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(np.int).tolist()
            
            if self.rect_size <= 40:
                scaled_rect_size = h // 3 if w > h else w // 3

                p1 = (self.x - self.rect_size, self.y - self.rect_size)
                p2 = (self.x + self.rect_size, self.y - self.rect_size)
                p3 = (self.x - self.rect_size, self.y + self.rect_size)
                
                wh = h if h < w else w                
                np1 = (w / 2 - wh / 4, h / 2 - wh / 4)
                np2 = (w / 2 + wh / 4, h / 2 - wh / 4)
                np3 = (w / 2 - wh / 4, h / 2 + wh / 4)
                
                mat = cv2.getAffineTransform( np.float32([p1,p2,p3])*self.view_scale, np.float32([np1,np2,np3]) )
                image = cv2.warpAffine(image, mat,(w,h) )                
                view_landmarks = LandmarksProcessor.transform_points (view_landmarks, mat)
     
            LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.image_size, self.face_type)

            if self.rect_locked:
                LandmarksProcessor.draw_landmarks(image, view_landmarks, (255,255,0) )
            self.redraw_needed = False
            
            io.show_image (self.wnd_name, image)
        else:
            if self.type == 'rects':
                self.result.append ( result )
            elif self.type == 'landmarks':
                self.result.append ( result )                        
            elif self.type == 'final':
                self.result += result
                         
            io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result


class DeletedFilesSearcherSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.debug_paths_stems = client_dict['debug_paths_stems']
            return None
            
        #override
        def process_data(self, data):  
            input_path_stem = Path(data[0]).stem        
            return any ( [ input_path_stem == d_stem for d_stem in self.debug_paths_stems] )
            
        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]
            
    #override
    def __init__(self, input_paths, debug_paths ): 
        self.input_paths = input_paths
        self.debug_paths_stems = [ Path(d).stem for d in debug_paths]        
        self.result = []
        super().__init__('DeletedFilesSearcherSubprocessor', DeletedFilesSearcherSubprocessor.Cli, 60)           
        
    #override
    def process_info_generator(self):    
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % (i), {}, {'debug_paths_stems' : self.debug_paths_stems}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Searching deleted files", len (self.input_paths))
            
    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.input_paths) > 0:        
            return [self.input_paths.pop(0)]        
        return None
    
    #override
    def on_data_return (self, host_dict, data):
        self.input_paths.insert(0, data[0])   
     
    #override
    def on_result (self, host_dict, data, result):
        if result == False:
            self.result.append( data[0] )
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def main(input_dir,
         output_dir,
         debug=False,
         detector='mt',
         manual_fix=False,
         manual_output_debug_fix=False,
         manual_window_size=1368,
         image_size=256,
         face_type='full_face',
         device_args={}):
         
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    face_type = FaceType.fromString(face_type)
    
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')
    
    if output_path.exists():
        if not manual_output_debug_fix:
            for filename in Path_utils.get_image_paths(output_path):
                Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        
    if manual_output_debug_fix:
        debug = True
        detector = 'manual'
        io.log_info('Performing re-extract frames which were deleted from _debug directory.')
        
    input_path_image_paths = Path_utils.get_image_unique_filestem_paths(input_path, verbose_print_func=io.log_info)
    if debug:
        debug_output_path = Path(str(output_path) + '_debug')
        
        if manual_output_debug_fix:
            if not debug_output_path.exists():
                raise ValueError("%s not found " % ( str(debug_output_path) ))

            input_path_image_paths = DeletedFilesSearcherSubprocessor (input_path_image_paths, Path_utils.get_image_paths(debug_output_path) ).run()
            input_path_image_paths = sorted (input_path_image_paths)            
        else:
            if debug_output_path.exists():
                for filename in Path_utils.get_image_paths(debug_output_path):
                    Path(filename).unlink()
            else:
                debug_output_path.mkdir(parents=True, exist_ok=True)

    images_found = len(input_path_image_paths)
    faces_detected = 0
    if images_found != 0:    
        if detector == 'manual':
            io.log_info ('Performing manual extract...')
            extracted_faces = ExtractSubprocessor ([ (filename,[]) for filename in input_path_image_paths ], 'landmarks', image_size, face_type, debug, cpu_only=cpu_only, manual=True, manual_window_size=manual_window_size).run()
        else:
            io.log_info ('Performing 1st pass...')
            extracted_rects = ExtractSubprocessor ([ (x,) for x in input_path_image_paths ], 'rects', image_size, face_type, debug, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, detector=detector).run()
                
            io.log_info ('Performing 2nd pass...')
            extracted_faces = ExtractSubprocessor (extracted_rects, 'landmarks', image_size, face_type, debug, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).run()
                
            if manual_fix:
                io.log_info ('Performing manual fix...')
                
                if all ( np.array ( [ len(data[1]) > 0 for data in extracted_faces] ) == True ):
                    io.log_info ('All faces are detected, manual fix not needed.')
                else:
                    extracted_faces = ExtractSubprocessor (extracted_faces, 'landmarks', image_size, face_type, debug, manual=True, manual_window_size=manual_window_size).run()

        if len(extracted_faces) > 0:
            io.log_info ('Performing 3rd pass...')
            final_imgs_paths = ExtractSubprocessor (extracted_faces, 'final', image_size, face_type, debug, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, output_path=output_path).run()
            faces_detected = len(final_imgs_paths)
            
    io.log_info ('-------------------------')
    io.log_info ('Images found:        %d' % (images_found) )
    io.log_info ('Faces detected:      %d' % (faces_detected) )
    io.log_info ('-------------------------')