import traceback
import os
import sys
import time
import multiprocessing
from pathlib import Path
import numpy as np
import cv2
from utils import Path_utils
from utils.DFLJPG import DFLJPG
from utils import image_utils
from facelib import FaceType
import facelib 
from nnlib import nnlib

from utils.SubprocessorBase import SubprocessorBase
class ExtractSubprocessor(SubprocessorBase):

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
        super().__init__('Extractor', no_response_time_sec)           

    #override
    def onHostClientsInitialized(self):
        if self.manual == True:
            self.wnd_name = 'Manual pass'
            cv2.namedWindow(self.wnd_name)

            self.landmarks = None
            self.param_x = -1
            self.param_y = -1
            self.param_rect_size = -1
            self.param = {'x': 0, 'y': 0, 'rect_size' : 5, 'rect_locked' : False, 'redraw_needed' : False }
            
            def onMouse(event, x, y, flags, param):        
                if event == cv2.EVENT_MOUSEWHEEL:
                    mod = 1 if flags > 0 else -1            
                    param['rect_size'] = max (5, param['rect_size'] + 10*mod)
                elif event == cv2.EVENT_LBUTTONDOWN:
                    param['rect_locked'] = not param['rect_locked']
                    param['redraw_needed'] = True
                elif not param['rect_locked']:
                    param['x'] = x
                    param['y'] = y
                    
            cv2.setMouseCallback(self.wnd_name, onMouse, self.param)
    
    def get_devices_for_type (self, type, multi_gpu):
        if (type == 'rects' or type == 'landmarks'):
            if multi_gpu:
                devices = nnlib.device.getDevicesWithAtLeastTotalMemoryGB(2)
            
            if not multi_gpu or len(devices) == 0:
                devices = [nnlib.device.getBestDeviceIdx()]
                
            if len(devices) == 0:
                devices = [0]
                    
            devices = [ (idx, nnlib.device.getDeviceName(idx), nnlib.device.getDeviceVRAMTotalGb(idx) ) for idx in devices]

        elif type == 'final':
            devices = [ (i, 'CPU%d' % (i), 0 ) for i in range(0, multiprocessing.cpu_count()) ]
            
        return devices 
        
    #override
    def process_info_generator(self):
        base_dict = {'type' : self.type, 
                     'image_size': self.image_size, 
                     'face_type': self.face_type, 
                     'debug': self.debug, 
                     'output_dir': str(self.output_path), 
                     'detector': self.detector}
    
        if not self.cpu_only:
            for (device_idx, device_name, device_total_vram_gb) in self.get_devices_for_type(self.type, self.multi_gpu): 
                num_processes = 1
                if not self.manual and self.type == 'rects' and self.detector == 'mt':
                    num_processes = int ( max (1, device_total_vram_gb / 2) )
                    
                for i in range(0, num_processes ):
                    client_dict = base_dict.copy()
                    client_dict['device_idx'] = device_idx
                    client_dict['device_name'] = device_name if num_processes == 1 else '%s #%d' % (device_name,i)
                    client_dict['device_type'] = 'GPU'
                    
                    yield client_dict['device_name'], {}, client_dict
        else:
            num_processes = 1
            if not self.manual and self.type == 'rects' and self.detector == 'mt':
                num_processes = int ( max (1, multiprocessing.cpu_count() / 2 ) )
            
            for i in range(0, num_processes ):
                client_dict = base_dict.copy()
                client_dict['device_idx'] = 0
                client_dict['device_name'] = 'CPU' if num_processes == 1 else 'CPU #%d' % (i),
                client_dict['device_type'] = 'CPU'
                
                yield client_dict['device_name'], {}, client_dict
                    
    #override
    def get_no_process_started_message(self):
        if (self.type == 'rects' or self.type == 'landmarks'):
            print ( 'You have no capable GPUs. Try to close programs which can consume VRAM, and run again.')
        elif self.type == 'final':
            print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):
        return None
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_data)
        
    #override
    def onHostGetData(self, host_dict):
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
                        prev_rect = faces.pop()[0]
                        self.param['rect_locked'] = True
                        faces.clear()
                        self.param['rect_size'] = ( prev_rect[2] - prev_rect[0] ) / 2
                        self.param['x'] = ( ( prev_rect[0] + prev_rect[2] ) / 2 ) * self.view_scale
                        self.param['y'] = ( ( prev_rect[1] + prev_rect[3] ) / 2 ) * self.view_scale

                if len(faces) == 0:
                    self.original_image = cv2.imread(filename)
                    
                    (h,w,c) = self.original_image.shape
 
                    self.view_scale = 1.0 if self.manual_window_size == 0 else self.manual_window_size / (w if w > h else h)
                    self.original_image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)    
                    (h,w,c) = self.original_image.shape
                    
                    self.text_lines_img = (image_utils.get_draw_text_lines ( self.original_image, (0,0, self.original_image.shape[1], min(100, self.original_image.shape[0]) ),
                                                    [   'Match landmarks with face exactly. Click to confirm/unconfirm selection',
                                                        '[Enter] - confirm face landmarks and continue',
                                                        '[Space] - confirm as unmarked frame and continue',
                                                        '[Mouse wheel] - change rect',
                                                        '[,] [.]- prev frame, next frame',
                                                        '[Q] - skip remaining frames'
                                                    ], (1, 1, 1) )*255).astype(np.uint8)           

                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        
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
                            if self.param['rect_locked']:
                                faces.append ( [(self.rect), self.landmarks] )
                            is_frame_done = True
                            break
                        elif key == ord(',')  and len(self.result) > 0:
                            # Only save the face if the rect is still locked
                            if self.param['rect_locked']:
                                faces.append ( [(self.rect), self.landmarks] )
                            go_to_prev_frame = True
                            break
                        elif key == ord('q'):
                            skip_remaining = True
                            break
                            
                        new_param_x = np.clip (self.param['x'], 0, w-1) / self.view_scale
                        new_param_y = np.clip (self.param['y'], 0, h-1) / self.view_scale
                        new_param_rect_size = self.param['rect_size']                        
                                
                        if self.param_x != new_param_x or \
                           self.param_y != new_param_y or \
                           self.param_rect_size != new_param_rect_size or \
                           self.param['redraw_needed']:
                           
                            self.param_x = new_param_x
                            self.param_y = new_param_y
                            self.param_rect_size = new_param_rect_size

                            self.rect = ( int(self.param_x-self.param_rect_size), 
                                          int(self.param_y-self.param_rect_size), 
                                          int(self.param_x+self.param_rect_size), 
                                          int(self.param_y+self.param_rect_size) )
                                          
                            return [filename, [self.rect]]
                            
                else:
                    is_frame_done = True
                    
                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    self.inc_progress_bar(1)
                    self.param['redraw_needed'] = True
                    self.param['rect_locked'] = False
                elif go_to_prev_frame:
                    self.input_data.insert(0, self.result.pop() )
                    self.inc_progress_bar(-1)
                    allow_remark_faces = True
                    self.param['redraw_needed'] = True
                    self.param['rect_locked'] = False
                elif skip_remaining:
                    while len(self.input_data) > 0:
                        self.result.append( self.input_data.pop(0) )
                        self.inc_progress_bar(1)

        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        if not self.manual:
            self.input_data.insert(0, data)   
        
    #override
    def onClientInitialize(self, client_dict):
        self.safe_print ('Running on %s.' % (client_dict['device_name']) )
        self.type         = client_dict['type']
        self.image_size   = client_dict['image_size']
        self.face_type    = client_dict['face_type']
        self.device_idx   = client_dict['device_idx']
        self.cpu_only     = client_dict['device_type'] == 'CPU'
        self.output_path  = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None        
        self.debug        = client_dict['debug']
        self.detector     = client_dict['detector']

        self.e = None

        device_config = nnlib.DeviceConfig ( cpu_only=self.cpu_only, force_gpu_idx=self.device_idx, allow_growth=True)
        if self.type == 'rects':
            if self.detector is not None:
                if self.detector == 'mt':
                    nnlib.import_all (device_config)
                    self.e = facelib.MTCExtractor(nnlib.keras, nnlib.tf, nnlib.tf_sess)                            
                elif self.detector == 'dlib':
                    nnlib.import_dlib (device_config)
                    self.e = facelib.DLIBExtractor(nnlib.dlib)
                self.e.__enter__()

        elif self.type == 'landmarks':
            nnlib.import_all (device_config)
            self.e = facelib.LandmarksExtractor(nnlib.keras)
            self.e.__enter__()
            
        elif self.type == 'final':
            pass
        
        return None

    #override
    def onClientFinalize(self):
        if self.e is not None:
            self.e.__exit__()
        
    #override
    def onClientProcessData(self, data):
        filename_path = Path( data[0] )

        image = cv2.imread( str(filename_path) )
        if image is None:
            print ( 'Failed to extract %s, reason: cv2.imread() fail.' % ( str(filename_path) ) )
        else:
            if self.type == 'rects':
                rects = self.e.extract_from_bgr (image)
                return [str(filename_path), rects]

            elif self.type == 'landmarks':
                rects = data[1]   
                landmarks = self.e.extract_from_bgr (image, rects)                    
                return [str(filename_path), landmarks]

            elif self.type == 'final':     
                result = []
                faces = data[1]
                
                if self.debug:
                    debug_output_file = '{}{}'.format( str(Path(str(self.output_path) + '_debug') / filename_path.stem),  '.jpg')
                    debug_image = image.copy()
                    
                for (face_idx, face) in enumerate(faces):         
                    output_file = '{}_{}{}'.format(str(self.output_path / filename_path.stem), str(face_idx), '.jpg')
                    
                    rect = face[0]
                    image_landmarks = np.array(face[1])

                    if self.debug:
                        facelib.LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, self.image_size, self.face_type)

                    if self.face_type == FaceType.MARK_ONLY:                        
                        face_image = image
                        face_image_landmarks = image_landmarks
                    else:
                        image_to_face_mat = facelib.LandmarksProcessor.get_transform_mat (image_landmarks, self.image_size, self.face_type)       
                        face_image = cv2.warpAffine(image, image_to_face_mat, (self.image_size, self.image_size), cv2.INTER_LANCZOS4)
                        face_image_landmarks = facelib.LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)
                    
                    cv2.imwrite(output_file, face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85] )

                    DFLJPG.embed_data(output_file, face_type = FaceType.toString(self.face_type),
                                                   landmarks = face_image_landmarks.tolist(),
                                                   yaw_value = facelib.LandmarksProcessor.calc_face_yaw (face_image_landmarks),
                                                   pitch_value = facelib.LandmarksProcessor.calc_face_pitch (face_image_landmarks),
                                                   source_filename = filename_path.name,
                                                   source_rect=  rect,
                                                   source_landmarks = image_landmarks.tolist()
                                        )  
                        
                    result.append (output_file)
                    
                if self.debug:
                    cv2.imwrite(debug_output_file, debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )
                    
                return result       
        return None

        #overridable
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[0]
        
    #override
    def onHostResult (self, host_dict, data, result):
        if self.manual == True:
            self.landmarks = result[1][0][1]
                                        
            image = cv2.addWeighted (self.original_image,1.0,self.text_lines_img,1.0,0)                    
            view_rect = (np.array(self.rect) * self.view_scale).astype(np.int).tolist()
            view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(np.int).tolist()
            facelib.LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.image_size, self.face_type)

            if self.param['rect_locked']:
                facelib.draw_landmarks(image, view_landmarks, (255,255,0) )
            self.param['redraw_needed'] = False
            
            cv2.imshow (self.wnd_name, image)
            return 0
        else:
            if self.type == 'rects':
                self.result.append ( result )
            elif self.type == 'landmarks':
                self.result.append ( result )                        
            elif self.type == 'final':
                self.result += result
                         
            return 1

    #override
    def onFinalizeAndGetResult(self):
        if self.manual == True:
            cv2.destroyAllWindows()
        return self.result

class DeletedFilesSearcherSubprocessor(SubprocessorBase):
    #override
    def __init__(self, input_paths, debug_paths ): 
        self.input_paths = input_paths
        self.debug_paths_stems = [ Path(d).stem for d in debug_paths]        
        self.result = []
        super().__init__('DeletedFilesSearcherSubprocessor', 60)           
        
    #override
    def process_info_generator(self):    
        for i in range(0, min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i), 
                                      'debug_paths_stems' : self.debug_paths_stems
                                      }

    #override
    def get_no_process_started_message(self):
        print ( 'Unable to start CPU processes.')
        
    #override
    def onHostGetProgressBarDesc(self):
        return "Searching deleted files"
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_paths)
        
    #override
    def onHostGetData(self, host_dict):
        if len (self.input_paths) > 0:        
            return [self.input_paths.pop(0)]        
        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        self.input_paths.insert(0, data[0])   
        
    #override
    def onClientInitialize(self, client_dict):
        self.debug_paths_stems = client_dict['debug_paths_stems']
        return None

    #override
    def onClientProcessData(self, data):  
        input_path_stem = Path(data[0]).stem        
        return any ( [ input_path_stem == d_stem for d_stem in self.debug_paths_stems] )

    #override
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data[0]
        
    #override
    def onHostResult (self, host_dict, data, result):
        if result == False:
            self.result.append( data[0] )
        return 1

    #override
    def onFinalizeAndGetResult(self):
        return self.result
        
'''
detector
    'dlib'
    'mt'
    'manual'

face_type
    'full_face'
    'avatar'
'''
def main (input_dir, output_dir, debug, detector='mt', multi_gpu=True, cpu_only=False, manual_fix=False, manual_output_debug_fix=False, manual_window_size=1368, image_size=256, face_type='full_face'):
    print ("Running extractor.\r\n")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    face_type = FaceType.fromString(face_type)

    if not input_path.exists():
        print('Input directory not found. Please ensure it exists.')
        return
        
    if output_path.exists():
        if not manual_output_debug_fix:
            for filename in Path_utils.get_image_paths(output_path):
                Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        
    if manual_output_debug_fix:
        debug = True
        detector = 'manual'
        print('Performing re-extract frames which were deleted from _debug directory.')
        
    input_path_image_paths = Path_utils.get_image_unique_filestem_paths(input_path, verbose=True)
    
    if debug:
        debug_output_path = Path(str(output_path) + '_debug')
        
        if manual_output_debug_fix:
            if not debug_output_path.exists():
                print ("%s not found " % ( str(debug_output_path) ))
                return
            
            input_path_image_paths = DeletedFilesSearcherSubprocessor ( input_path_image_paths, Path_utils.get_image_paths(debug_output_path) ).process()
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
            print ('Performing manual extract...')
            extracted_faces = ExtractSubprocessor ([ (filename,[]) for filename in input_path_image_paths ], 'landmarks', image_size, face_type, debug, cpu_only=cpu_only, manual=True, manual_window_size=manual_window_size).process()
        else:
            print ('Performing 1st pass...')
            extracted_rects = ExtractSubprocessor ([ (x,) for x in input_path_image_paths ], 'rects', image_size, face_type, debug, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, detector=detector).process()
                
            print ('Performing 2nd pass...')
            extracted_faces = ExtractSubprocessor (extracted_rects, 'landmarks', image_size, face_type, debug, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).process()
                
            if manual_fix:
                print ('Performing manual fix...')
                
                if all ( np.array ( [ len(data[1]) > 0 for data in extracted_faces] ) == True ):
                    print ('All faces are detected, manual fix not needed.')
                else:
                    extracted_faces = ExtractSubprocessor (extracted_faces, 'landmarks', image_size, face_type, debug, manual=True, manual_window_size=manual_window_size).process()

        if len(extracted_faces) > 0:
            print ('Performing 3rd pass...')
            final_imgs_paths = ExtractSubprocessor (extracted_faces, 'final', image_size, face_type, debug, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, output_path=output_path).process()
            faces_detected = len(final_imgs_paths)
            
    print('-------------------------')
    print('Images found:        %d' % (images_found) )
    print('Faces detected:      %d' % (faces_detected) )
    print('-------------------------')