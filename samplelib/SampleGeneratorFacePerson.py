import multiprocessing
import traceback

import cv2
import numpy as np

from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)
from utils import iter_utils


'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFacePerson(SampleGeneratorBase):
    def __init__ (self, samples_path, debug=False, batch_size=1, 
                        sample_process_options=SampleProcessor.Options(), 
                        output_sample_types=[], 
                        person_id_mode=1,
                        use_caching=False,
                        generators_count=2, 
                        generators_random_seed=None,
                        **kwargs):
                        
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.person_id_mode = person_id_mode

        if generators_random_seed is not None and len(generators_random_seed) != generators_count:
            raise ValueError("len(generators_random_seed) != generators_count")
        self.generators_random_seed = generators_random_seed
        
        samples = SampleLoader.load (SampleType.FACE, self.samples_path, person_id_mode=True, use_caching=use_caching)
         
        if person_id_mode==1:
            np.random.shuffle(samples)
            
            new_samples = []
            while len(samples) > 0:
                for i in range( len(samples)-1, -1, -1):
                    sample = samples[i]
                    
                    if len(sample) > 0:
                        new_samples.append(sample.pop(0))
                        
                    if len(sample) == 0:
                        samples.pop(i)
            samples = new_samples
            #new_samples = []
            #for s in samples:    
            #    new_samples += s
            #samples = new_samples
            #np.random.shuffle(samples)
            
        self.samples_len = len(samples)
        
        if self.samples_len == 0:
            raise ValueError('No training data provided.')        

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (0, samples) )]
        else:
            self.generators_count = min ( generators_count, self.samples_len )
            
            if person_id_mode==1:
                self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (i, samples[i::self.generators_count]) ) for i in range(self.generators_count) ]
            else:
                self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (i, samples) ) for i in range(self.generators_count) ]

        self.generator_counter = -1
    
    #overridable
    def get_total_sample_count(self):
        return self.samples_len
        
    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):        
        generator_id, samples = param

        if self.generators_random_seed is not None:
            np.random.seed ( self.generators_random_seed[generator_id] )

        if self.person_id_mode==1:
            samples_len = len(samples)
            samples_idxs = [*range(samples_len)]
            shuffle_idxs = []
        elif self.person_id_mode==2:
            persons_count = len(samples)
            
            person_idxs = []
            for j in range(persons_count):
                for i in range(j+1,persons_count):
                    person_idxs += [ [i,j] ]

            shuffle_person_idxs = []
            
            samples_idxs = [None]*persons_count
            shuffle_idxs = [None]*persons_count
            
            for i in range(persons_count):
                samples_idxs[i] = [*range(len(samples[i]))]
                shuffle_idxs[i] = []

        while True:           
            
            if self.person_id_mode==2: 
                if len(shuffle_person_idxs) == 0:
                    shuffle_person_idxs = person_idxs.copy()
                    np.random.shuffle(shuffle_person_idxs)
                person_ids = shuffle_person_idxs.pop()
            
                        
            batches = None
            for n_batch in range(self.batch_size):
   
                if self.person_id_mode==1:
                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = samples_idxs.copy()
                        #np.random.shuffle(shuffle_idxs)

                    idx = shuffle_idxs.pop()
                    sample = samples[ idx ]
    
                    try:
                        x = SampleProcessor.process (sample, self.sample_process_options, self.output_sample_types, self.debug)
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                    if type(x) != tuple and type(x) != list:
                        raise Exception('SampleProcessor.process returns NOT tuple/list')

                    if batches is None:
                        batches = [ [] for _ in range(len(x)) ]
                        
                        batches += [ [] ]
                        i_person_id = len(batches)-1

                    for i in range(len(x)):
                        batches[i].append ( x[i] )

                    batches[i_person_id].append ( np.array([sample.person_id]) )

                    
                else:
                    person_id1, person_id2 = person_ids
                    
                    if len(shuffle_idxs[person_id1]) == 0:
                        shuffle_idxs[person_id1] = samples_idxs[person_id1].copy()
                        np.random.shuffle(shuffle_idxs[person_id1])

                    idx = shuffle_idxs[person_id1].pop()
                    sample1 = samples[person_id1][idx]
                    
                    if len(shuffle_idxs[person_id2]) == 0:
                        shuffle_idxs[person_id2] = samples_idxs[person_id2].copy()
                        np.random.shuffle(shuffle_idxs[person_id2])

                    idx = shuffle_idxs[person_id2].pop()
                    sample2 = samples[person_id2][idx]
                
                    if sample1 is not None and sample2 is not None:
                        try:
                            x1 = SampleProcessor.process (sample1, self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample1.filename, traceback.format_exc() ) )
                        
                        try:
                            x2 = SampleProcessor.process (sample2, self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample2.filename, traceback.format_exc() ) )

                        x1_len = len(x1)
                        if batches is None:
                            batches = [ [] for _ in range(x1_len) ]                            
                            batches += [ [] ]
                            i_person_id1 = len(batches)-1
                            
                            batches += [ [] for _ in range(len(x2)) ]                            
                            batches += [ [] ]
                            i_person_id2 = len(batches)-1

                        for i in range(x1_len):
                            batches[i].append ( x1[i] )
                            
                        for i in range(len(x2)):
                            batches[x1_len+1+i].append ( x2[i] )

                        batches[i_person_id1].append ( np.array([sample1.person_id]) )

                        batches[i_person_id2].append ( np.array([sample2.person_id]) )
                        
                    

            yield [ np.array(batch) for batch in batches]
    
    @staticmethod
    def get_person_id_max_count(samples_path):
        return SampleLoader.get_person_id_max_count(samples_path)