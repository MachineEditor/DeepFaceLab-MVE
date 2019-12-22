import copy
import multiprocessing
import traceback

import cv2
import numpy as np

from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleHost, SampleProcessor,
                       SampleType)
from utils import iter_utils, mp_utils


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
                        **kwargs):
                        
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.person_id_mode = person_id_mode


        samples_host = SampleHost.mp_host (SampleType.FACE, self.samples_path)
        samples = samples_host.get_list()
        self.samples_len = len(samples)

        if self.samples_len == 0:
            raise ValueError('No training data provided.')
        
        persons_name_idxs = {}
        
        for i,sample in enumerate(samples):
            person_name = sample.person_name
            if person_name not in persons_name_idxs:
                persons_name_idxs[person_name] = []                
            persons_name_idxs[person_name].append (i)
  
        indexes2D = [ persons_name_idxs[person_name] for person_name in sorted(list(persons_name_idxs.keys())) ]
        index2d_host = mp_utils.Index2DHost(indexes2D)
        

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (samples_host.create_cli(), index2d_host.create_cli(),) )]
        else:
            self.generators_count = np.clip(multiprocessing.cpu_count(), 2, 4)
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (samples_host.create_cli(), index2d_host.create_cli(),), start_now=True ) for i in range(self.generators_count) ]

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
        samples, index2d_host, = param
        bs = self.batch_size

        while True:
            person_idxs = index2d_host.get_1D(bs)            
            samples_idxs = index2d_host.get_2D(person_idxs, 1)
            
            batches = None
            for n_batch in range(bs):
                person_id = person_idxs[n_batch]
                sample_idx = samples_idxs[n_batch][0]

                sample = samples[ sample_idx ]
                try:
                    x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)
                except:
                    raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )
  
                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]
                    
                    batches += [ [] ]
                    i_person_id = len(batches)-1

                for i in range(len(x)):
                    batches[i].append ( x[i] )

                batches[i_person_id].append ( np.array([person_id]) )
            
            yield [ np.array(batch) for batch in batches]
    
    @staticmethod
    def get_person_id_max_count(samples_path):
        return SampleHost.get_person_id_max_count(samples_path)

"""
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
        elif self.person_id_mode==3:
            persons_count = len(samples)
            
            person_idxs = [ *range(persons_count) ]
            shuffle_person_idxs = []
            
            samples_idxs = [None]*persons_count
            shuffle_idxs = [None]*persons_count
            
            for i in range(persons_count):
                samples_idxs[i] = [*range(len(samples[i]))]
                shuffle_idxs[i] = []
                
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
                        np.random.shuffle(shuffle_idxs) ###

                    idx = shuffle_idxs.pop()
                    sample = samples[ idx ]
    
                    try:
                        x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)
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

                    
                elif self.person_id_mode==2:
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
                            x1, = SampleProcessor.process ([sample1], self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample1.filename, traceback.format_exc() ) )
                        
                        try:
                            x2, = SampleProcessor.process ([sample2], self.sample_process_options, self.output_sample_types, self.debug)
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
                        
                elif self.person_id_mode==3:             
                    if len(shuffle_person_idxs) == 0:
                        shuffle_person_idxs = person_idxs.copy()
                        np.random.shuffle(shuffle_person_idxs)
                    person_id = shuffle_person_idxs.pop()
                       
                    if len(shuffle_idxs[person_id]) == 0:
                        shuffle_idxs[person_id] = samples_idxs[person_id].copy()
                        np.random.shuffle(shuffle_idxs[person_id])

                    idx = shuffle_idxs[person_id].pop()
                    sample1 = samples[person_id][idx]
                    
                    if len(shuffle_idxs[person_id]) == 0:
                        shuffle_idxs[person_id] = samples_idxs[person_id].copy()
                        np.random.shuffle(shuffle_idxs[person_id])

                    idx = shuffle_idxs[person_id].pop()
                    sample2 = samples[person_id][idx]
                
                    if sample1 is not None and sample2 is not None:
                        try:
                            x1, = SampleProcessor.process ([sample1], self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample1.filename, traceback.format_exc() ) )
                        
                        try:
                            x2, = SampleProcessor.process ([sample2], self.sample_process_options, self.output_sample_types, self.debug)
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
"""