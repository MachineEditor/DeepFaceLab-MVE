import traceback
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from utils.AlignedPNG import AlignedPNG
from utils import iter_utils
from utils import Path_utils
from .BaseTypes import TrainingDataType
from .BaseTypes import TrainingDataSample
from facelib import FaceType
from facelib import LandmarksProcessor

'''
You can implement your own TrainingDataGenerator
'''
class TrainingDataGeneratorBase(object):
    cache = dict()
    
    #DONT OVERRIDE
    #use YourOwnTrainingDataGenerator (..., your_opt=1)
    #and then this opt will be passed in YourOwnTrainingDataGenerator.onInitialize ( your_opt )
    def __init__ (self, trainingdatatype, training_data_path, target_training_data_path=None, debug=False, batch_size=1, **kwargs):
        if not isinstance(trainingdatatype, TrainingDataType):
            raise Exception('TrainingDataGeneratorBase() trainingdatatype is not TrainingDataType')

        if training_data_path is None:
            raise Exception('training_data_path is None')
            
        self.training_data_path = Path(training_data_path)
        self.target_training_data_path = Path(target_training_data_path) if target_training_data_path is not None else None

        self.debug = debug
        self.batch_size = 1 if self.debug else batch_size        
        self.trainingdatatype = trainingdatatype
        self.data = TrainingDataGeneratorBase.load (trainingdatatype, self.training_data_path, self.target_training_data_path)        

        if self.debug:
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, self.data)]
        else:
            if len(self.data) > 1:
                self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, self.data[0::2] ),
                                   iter_utils.SubprocessGenerator ( self.batch_func, self.data[1::2] )]
            else:
                self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, self.data )]
                
        self.generator_counter = -1            
        self.onInitialize(**kwargs)
        
    #overridable
    def onInitialize(self, **kwargs):
        #your TrainingDataGenerator initialization here
        pass
        
    #overridable
    def onProcessSample(self, sample, debug):
        #process sample and return tuple of images for your model in onTrainOneEpoch
        return ( np.zeros( (64,64,4), dtype=np.float32 ), )
        
    def __iter__(self):
        return self
        
    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        x = next(generator) 
        return x
        
    def batch_func(self, data):    
        data_len = len(data)
        if data_len == 0:
            raise ValueError('No training data provided.')
            
        if self.trainingdatatype == TrainingDataType.FACE_YAW_SORTED or self.trainingdatatype == TrainingDataType.FACE_YAW_SORTED_AS_TARGET:
            if all ( [ x == None for x in data] ):
                raise ValueError('Not enough training data. Gather more faces!')
             
        if self.trainingdatatype == TrainingDataType.IMAGE or self.trainingdatatype == TrainingDataType.FACE:
            shuffle_idxs = []          
        elif self.trainingdatatype == TrainingDataType.FACE_YAW_SORTED or self.trainingdatatype == TrainingDataType.FACE_YAW_SORTED_AS_TARGET:
            shuffle_idxs = []            
            shuffle_idxs_2D = [[]]*data_len
            
        while True:                
            
            batches = None
            for n_batch in range(0, self.batch_size):
                while True:
                    sample = None
                                
                    if self.trainingdatatype == TrainingDataType.IMAGE or self.trainingdatatype == TrainingDataType.FACE:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = [ i for i in range(0, data_len) ]
                            random.shuffle(shuffle_idxs)                            
                        idx = shuffle_idxs.pop()
                        sample = data[ idx ]
                    elif self.trainingdatatype == TrainingDataType.FACE_YAW_SORTED or self.trainingdatatype == TrainingDataType.FACE_YAW_SORTED_AS_TARGET:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = [ i for i in range(0, data_len) ]
                            random.shuffle(shuffle_idxs)
                        
                        idx = shuffle_idxs.pop()                        
                        if data[idx] != None:
                            if len(shuffle_idxs_2D[idx]) == 0:
                                shuffle_idxs_2D[idx] = [ i for i in range(0, len(data[idx])) ]
                                random.shuffle(shuffle_idxs_2D[idx])
                                
                            idx2 = shuffle_idxs_2D[idx].pop()                            
                            sample = data[idx][idx2]
                            
                    if sample is not None:          
                        try:
                            x = self.onProcessSample (sample, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )
                        
                        if type(x) != tuple and type(x) != list:
                            raise Exception('TrainingDataGenerator.onProcessSample() returns NOT tuple/list')
                            
                        x_len = len(x)
                        if batches is None:
                            batches = [ [] for _ in range(0,x_len) ]
                            
                        for i in range(0,x_len):
                            batches[i].append ( x[i] )
                            
                        break
                        
            yield [ np.array(batch) for batch in batches]
        
    def get_dict_state(self):
        return {}

    def set_dict_state(self, state):
        pass

    @staticmethod
    def load(trainingdatatype, training_data_path, target_training_data_path=None):
        cache = TrainingDataGeneratorBase.cache
        
        if str(training_data_path) not in cache.keys():
            cache[str(training_data_path)] = [None]*TrainingDataType.QTY
            
        if target_training_data_path is not None and str(target_training_data_path) not in cache.keys():
            cache[str(target_training_data_path)] = [None]*TrainingDataType.QTY
            
        datas = cache[str(training_data_path)]

        if            trainingdatatype == TrainingDataType.IMAGE:
            if  datas[trainingdatatype] is None:  
                datas[trainingdatatype] = [ TrainingDataSample(filename=filename) for filename in tqdm( Path_utils.get_image_paths(training_data_path), desc="Loading" ) ]

        elif          trainingdatatype == TrainingDataType.FACE:
            if  datas[trainingdatatype] is None:  
                datas[trainingdatatype] = X_LOAD( [ TrainingDataSample(filename=filename) for filename in Path_utils.get_image_paths(training_data_path) ] )
        
        elif          trainingdatatype == TrainingDataType.FACE_YAW_SORTED:
            if  datas[trainingdatatype] is None:
                datas[trainingdatatype] = X_YAW_SORTED( TrainingDataGeneratorBase.load(TrainingDataType.FACE, training_data_path) )
        
        elif          trainingdatatype == TrainingDataType.FACE_YAW_SORTED_AS_TARGET:            
            if  datas[trainingdatatype] is None:
                if target_training_data_path is None:
                    raise Exception('target_training_data_path is None for FACE_YAW_SORTED_AS_TARGET')
                datas[trainingdatatype] = X_YAW_AS_Y_SORTED( TrainingDataGeneratorBase.load(TrainingDataType.FACE_YAW_SORTED, training_data_path), TrainingDataGeneratorBase.load(TrainingDataType.FACE_YAW_SORTED, target_training_data_path) )
            
        return datas[trainingdatatype]
        
def X_LOAD ( RAWS ):
    sample_list = []
    
    for s in tqdm( RAWS, desc="Loading" ):

        s_filename_path = Path(s.filename)
        if s_filename_path.suffix != '.png':
            print ("%s is not a png file required for training" % (s_filename_path.name) ) 
            continue
        
        a_png = AlignedPNG.load ( str(s_filename_path) )
        if a_png is None:
            print ("%s failed to load" % (s_filename_path.name) )
            continue

        d = a_png.getFaceswapDictData()
        if d is None or d['landmarks'] is None or d['yaw_value'] is None:
            print ("%s - no embedded faceswap info found required for training" % (s_filename_path.name) ) 
            continue
            
        face_type = d['face_type'] if 'face_type' in d.keys() else 'full_face'        
        face_type = FaceType.fromString (face_type) 
        sample_list.append( s.copy_and_set(face_type=face_type, shape=a_png.get_shape(), landmarks=d['landmarks'], yaw=d['yaw_value']) )
        
    return sample_list
    
def X_YAW_SORTED( YAW_RAWS ):

    lowest_yaw, highest_yaw = -32, +32      
    gradations = 64
    diff_rot_per_grad = abs(highest_yaw-lowest_yaw) / gradations

    yaws_sample_list = [None]*gradations
    
    for i in tqdm( range(0, gradations), desc="Sorting" ):
        yaw = lowest_yaw + i*diff_rot_per_grad
        next_yaw = lowest_yaw + (i+1)*diff_rot_per_grad

        yaw_samples = []        
        for s in YAW_RAWS:                
            s_yaw = s.yaw
            if (i == 0            and s_yaw < next_yaw) or \
               (i  < gradations-1 and s_yaw >= yaw and s_yaw < next_yaw) or \
               (i == gradations-1 and s_yaw >= yaw):
                yaw_samples.append ( s )
                
        if len(yaw_samples) > 0:
            yaws_sample_list[i] = yaw_samples
    
    return yaws_sample_list
    
def X_YAW_AS_Y_SORTED (s, t):
    l = len(s)
    if l != len(t):
        raise Exception('X_YAW_AS_Y_SORTED() s_len != t_len')
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
                new_s[t_idx] = [ sample.copy_and_set(mirror=True, yaw=-sample.yaw, landmarks=LandmarksProcessor.mirror_landmarks (sample.landmarks, sample.shape[1] ))
                                      for sample in s[search_idx] 
                                    ] if mirrored else s[search_idx]                
                break
             
    return new_s
