import traceback
import numpy as np
import random
import cv2

from utils import iter_utils

from samples import SampleType
from samples import SampleProcessor
from samples import SampleLoader
from samples import SampleGeneratorBase

'''
arg 
output_sample_types = [ 
                        [SampleProcessor.TypeFlags, size, (optional)random_sub_size] , 
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, sort_by_yaw=False, sort_by_yaw_target_samples_path=None, with_close_to_self=False, sample_process_options=SampleProcessor.Options(), output_sample_types=[], generators_count=2, **kwargs):
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
               
        if sort_by_yaw_target_samples_path is not None:
            self.sample_type = SampleType.FACE_YAW_SORTED_AS_TARGET
        elif sort_by_yaw:
            self.sample_type = SampleType.FACE_YAW_SORTED
        elif with_close_to_self:
            self.sample_type = SampleType.FACE_WITH_CLOSE_TO_SELF
        else:
            self.sample_type = SampleType.FACE         
  
        self.samples = SampleLoader.load (self.sample_type, self.samples_path, sort_by_yaw_target_samples_path)        

        self.generators_count = min ( generators_count, len(self.samples) )

        if self.debug:
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, 0 )]
        else:
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, i ) for i in range(self.generators_count) ]
  
        self.generator_counter = -1
    
    def __iter__(self):
        return self
        
    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)
        
    def batch_func(self, generator_id):    
        samples = self.samples[generator_id::self.generators_count]
        
        data_len = len(samples)
        if data_len == 0:
            raise ValueError('No training data provided.')
            
        if self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            if all ( [ x == None for x in samples] ):
                raise ValueError('Not enough training data. Gather more faces!')
             
        if self.sample_type == SampleType.FACE or self.sample_type == SampleType.FACE_WITH_CLOSE_TO_SELF:
            shuffle_idxs = []          
        elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            shuffle_idxs = []            
            shuffle_idxs_2D = [[]]*data_len
            
        while True:                
            
            batches = None
            for n_batch in range(self.batch_size):
                while True:
                    sample = None
                                
                    if self.sample_type == SampleType.FACE or self.sample_type == SampleType.FACE_WITH_CLOSE_TO_SELF:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = random.sample( range(data_len), data_len )
                        idx = shuffle_idxs.pop()
                        sample = samples[ idx ]
                    elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = random.sample( range(data_len), data_len )
                        
                        idx = shuffle_idxs.pop()                        
                        if samples[idx] != None:
                            if len(shuffle_idxs_2D[idx]) == 0:
                                shuffle_idxs_2D[idx] = random.sample( range(len(samples[idx])), len(samples[idx]) )
                                
                            idx2 = shuffle_idxs_2D[idx].pop()                            
                            sample = samples[idx][idx2]
                            
                    if sample is not None:          
                        try:
                            x = SampleProcessor.process (sample, self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )
                        
                        if type(x) != tuple and type(x) != list:
                            raise Exception('SampleProcessor.process returns NOT tuple/list')
                            
                        if batches is None:
                            batches = [ [] for _ in range(len(x)) ]
                            
                        for i in range(len(x)):
                            batches[i].append ( x[i] )
                            
                        break
                        
            yield [ np.array(batch) for batch in batches]
