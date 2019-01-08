import sys
import os
import traceback
from pathlib import Path
from utils import Path_utils
import cv2
from tqdm import tqdm
from utils.DFLPNG import DFLPNG
from utils import image_utils
import shutil
import numpy as np
import time
import multiprocessing
from models import ConverterBase

class model_process_predictor(object):
    def __init__(self, sq, cq, lock):
        self.sq = sq
        self.cq = cq
        self.lock = lock
        
    def __call__(self, face):
        self.lock.acquire()
        
        self.sq.put ( {'op': 'predict', 'face' : face} )
        while True:
            if not self.cq.empty():
                obj = self.cq.get()
                obj_op = obj['op']
                if obj_op == 'predict_result':
                    self.lock.release()
                    return obj['result']
            time.sleep(0.005)
        
def model_process(stdin_fd, model_name, model_dir, in_options, sq, cq):
    sys.stdin = os.fdopen(stdin_fd)
    
    try:    
        model_path = Path(model_dir)
        
        import models 
        model = models.import_model(model_name)(model_path, **in_options)
        converter = model.get_converter(**in_options)
        converter.dummy_predict()
        
        cq.put ( {'op':'init', 'converter' : converter.copy_and_set_predictor( None ) } )

        while True:
            while not sq.empty():
                obj = sq.get()
                obj_op = obj['op']
                if obj_op == 'predict':
                    result = converter.predictor ( obj['face'] )
                    cq.put ( {'op':'predict_result', 'result':result} )
            time.sleep(0.005)        
    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()
    
    
    
from utils.SubprocessorBase import SubprocessorBase
class ConvertSubprocessor(SubprocessorBase):

    #override
    def __init__(self, converter, input_path_image_paths, output_path, alignments, debug = False, **in_options): 
        super().__init__('Converter', 86400 if debug == True else 60)    
        self.converter = converter
        self.input_path_image_paths = input_path_image_paths
        self.output_path = output_path
        self.alignments = alignments
        self.debug = debug
        self.in_options = in_options
        self.input_data = self.input_path_image_paths
        self.files_processed = 0
        self.faces_processed = 0
        
    #override
    def process_info_generator(self):
        r = [0] if self.debug else range( min(multiprocessing.cpu_count(), 6) )
        
        

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i), 
                                      'converter' : self.converter, 
                                      'output_dir' : str(self.output_path), 
                                      'alignments' : self.alignments,
                                      'debug': self.debug,
                                      'in_options': self.in_options
                                      }
     
    #override
    def get_no_process_started_message(self):
        return 'Unable to start CPU processes.'
        
    #override
    def onHostGetProgressBarDesc(self):
        return "Converting"
        
    #override
    def onHostGetProgressBarLen(self):
        return len (self.input_data)
        
    #override
    def onHostGetData(self, host_dict):
        if len (self.input_data) > 0:
            return self.input_data.pop(0)            
        return None
    
    #override
    def onHostDataReturn (self, host_dict, data):
        self.input_data.insert(0, data)   
        
    #overridable
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return data
        
    #override
    def onClientInitialize(self, client_dict):
        print ('Running on %s.' % (client_dict['device_name']) )
        self.device_idx  = client_dict['device_idx']
        self.device_name = client_dict['device_name']
        self.converter   = client_dict['converter']
        self.output_path = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None        
        self.alignments  = client_dict['alignments']
        self.debug       = client_dict['debug']
        
        from nnlib import nnlib          
        #model process ate all GPU mem,
        #so we cannot use GPU for any TF operations in converter processes (for example image_utils.TFLabConverter)
        #therefore forcing active_DeviceConfig to CPU only
        nnlib.active_DeviceConfig = nnlib.DeviceConfig (cpu_only=True)
        
        return None

    #override
    def onClientFinalize(self):
        pass
        
    #override
    def onClientProcessData(self, data):
        filename_path = Path(data)

        files_processed = 1
        faces_processed = 0
            
        output_filename_path = self.output_path / filename_path.name
        if self.converter.get_mode() == ConverterBase.MODE_FACE and filename_path.stem not in self.alignments.keys():                    
            if not self.debug:
                print ( 'no faces found for %s, copying without faces' % (filename_path.name) )
                shutil.copy ( str(filename_path), str(output_filename_path) )
        else:
            image = (cv2.imread(str(filename_path)) / 255.0).astype(np.float32)

            if self.converter.get_mode() == ConverterBase.MODE_IMAGE:
                image = self.converter.convert_image(image, None, self.debug)
                if self.debug:
                    for img in image:
                        cv2.imshow ('Debug convert', img )
                        cv2.waitKey(0)
                faces_processed = 1
            elif self.converter.get_mode() == ConverterBase.MODE_IMAGE_WITH_LANDMARKS:
                image_landmarks = DFLPNG.load( str(filename_path), throw_on_no_embedded_data=True ).get_landmarks()
                        
                image = self.converter.convert_image(image, image_landmarks, self.debug)
                if self.debug:
                    for img in image:
                        cv2.imshow ('Debug convert', img )
                        cv2.waitKey(0)
                faces_processed = 1
            elif self.converter.get_mode() == ConverterBase.MODE_FACE:
                faces = self.alignments[filename_path.stem]
                for face_num, image_landmarks in enumerate(faces):
                    try:
                        if self.debug:
                            print ( '\nConverting face_num [%d] in file [%s]' % (face_num, filename_path) )
                        
                        image = self.converter.convert_face(image, image_landmarks, self.debug)     
                        if self.debug:
                            for img in image:
                                cv2.imshow ('Debug convert', img )
                                cv2.waitKey(0)
                    except Exception as e:
                        print ( 'Error while converting face_num [%d] in file [%s]: %s' % (face_num, filename_path, str(e)) )
                        traceback.print_exc()
                faces_processed = len(faces)
                    
            if not self.debug:
                cv2.imwrite (str(output_filename_path), (image*255).astype(np.uint8) )
            
            
        return (files_processed, faces_processed)
        
    #override
    def onHostResult (self, host_dict, data, result):
        self.files_processed += result[0]
        self.faces_processed += result[1]    
        return 1
             
    #override
    def onFinalizeAndGetResult(self):
        return self.files_processed, self.faces_processed
        
def main (input_dir, output_dir, model_dir, model_name, aligned_dir=None, **in_options):
    print ("Running converter.\r\n")
    
    debug = in_options['debug']
    
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)        
        model_path = Path(model_dir)
        
        if not input_path.exists():
            print('Input directory not found. Please ensure it exists.')
            return

        if output_path.exists():
            for filename in Path_utils.get_image_paths(output_path):
                Path(filename).unlink()
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            
        
            
        if not model_path.exists():
            print('Model directory not found. Please ensure it exists.')
            return
   
        model_sq = multiprocessing.Queue()
        model_cq = multiprocessing.Queue()
        model_lock = multiprocessing.Lock()
        model_p = multiprocessing.Process(target=model_process, args=( sys.stdin.fileno(), model_name, model_dir, in_options, model_sq, model_cq))
        model_p.start()
        
        while True:
            if not model_cq.empty():
                obj = model_cq.get()
                obj_op = obj['op']
                if obj_op == 'init':
                    converter = obj['converter']
                    break

        alignments = None
        
        if converter.get_mode() == ConverterBase.MODE_FACE:            
            if aligned_dir is None:
                print('Aligned directory not found. Please ensure it exists.')
                return 
                
            aligned_path = Path(aligned_dir)
            if not aligned_path.exists():
                print('Aligned directory not found. Please ensure it exists.')
                return 
                
            alignments = {}
            
            aligned_path_image_paths = Path_utils.get_image_paths(aligned_path)
            for filename in tqdm(aligned_path_image_paths, desc= "Collecting alignments" ):
                dflpng = DFLPNG.load( str(filename), print_on_no_embedded_data=True )                
                if dflpng is None:
                    continue
                
                source_filename_stem = Path( dflpng.get_source_filename() ).stem
                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments[ source_filename_stem ].append (dflpng.get_source_landmarks())
        
            
            #interpolate landmarks
            #from facelib import LandmarksProcessor
            #from facelib import FaceType
            #a = sorted(alignments.keys())
            #a_len = len(a)
            #
            #box_pts = 3            
            #box = np.ones(box_pts)/box_pts
            #for i in range( a_len ):
            #    if i >= box_pts and i <= a_len-box_pts-1:
            #        af0 = alignments[ a[i] ][0] ##first face
            #        m0 = LandmarksProcessor.get_transform_mat (af0, 256, face_type=FaceType.FULL)      
            #    
            #        points = []
            #        
            #        for j in range(-box_pts, box_pts+1):
            #            af = alignments[ a[i+j] ][0] ##first face
            #            m = LandmarksProcessor.get_transform_mat (af, 256, face_type=FaceType.FULL)                    
            #            p = LandmarksProcessor.transform_points (af, m)
            #            points.append (p)
            #    
            #        points = np.array(points)
            #        points_len = len(points)
            #        t_points = np.transpose(points, [1,0,2])
            #        
            #        p1 = np.array ( [ int(np.convolve(x[:,0], box, mode='same')[points_len//2]) for x in t_points ] )
            #        p2 = np.array ( [ int(np.convolve(x[:,1], box, mode='same')[points_len//2]) for x in t_points ] )
            #        
            #        new_points = np.concatenate( [np.expand_dims(p1,-1),np.expand_dims(p2,-1)], -1 )
            #        
            #        alignments[ a[i] ][0]  = LandmarksProcessor.transform_points (new_points, m0, True).astype(np.int32)
            
        files_processed, faces_processed = ConvertSubprocessor ( 
                    converter              = converter.copy_and_set_predictor( model_process_predictor(model_sq,model_cq,model_lock) ), 
                    input_path_image_paths = Path_utils.get_image_paths(input_path), 
                    output_path            = output_path,
                    alignments             = alignments,  
                    **in_options ).process()

        model_p.terminate()
        
        '''            
        if model_name == 'AVATAR':
            output_path_image_paths = Path_utils.get_image_paths(output_path)
            
            last_ok_frame = -1
            for filename in output_path_image_paths:
                filename_path = Path(filename)
                stem = Path(filename).stem
                try:
                    frame = int(stem)
                except:
                    raise Exception ('Aligned avatars must be created from indexed sequence files.')
                    
                if frame-last_ok_frame > 1:
                    start = last_ok_frame + 1
                    end = frame - 1
                    
                    print ("Filling gaps: [%d...%d]" % (start, end) )
                    for i in range (start, end+1):                    
                        shutil.copy ( str(filename), str( output_path / ('%.5d%s' % (i, filename_path.suffix ))  ) )
                    
                last_ok_frame = frame
        '''
        
    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()
    
   
