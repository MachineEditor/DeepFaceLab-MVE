import pickle
import traceback

import cv2
import numpy as np

from samplelib import (SampleGeneratorBase, SampleHost, SampleProcessor,
                       SampleType)
from utils import iter_utils


'''
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFaceTemporal(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, temporal_image_count, sample_process_options=SampleProcessor.Options(), output_sample_types=[], generators_count=2, **kwargs):
        super().__init__(samples_path, debug, batch_size)

        self.temporal_image_count = temporal_image_count
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = generators_count

        samples = SampleHost.load (SampleType.FACE_TEMPORAL_SORTED, self.samples_path)
        samples_len = len(samples)
        if samples_len == 0:
            raise ValueError('No training data provided.')
        
        pickled_samples = pickle.dumps(samples, 4)
        if self.debug:
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (0, pickled_samples) )]
        else:
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (i, pickled_samples) ) for i in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param):
        generator_id, pickled_samples = param
        samples = pickle.loads(pickled_samples)
        samples_len = len(samples)
        
        mult_max = 1
        l = samples_len - ( (self.temporal_image_count)*mult_max - (mult_max-1)  )

        samples_idxs = [ *range(l+1) ]

        if len(samples_idxs) - self.temporal_image_count < 0:
            raise ValueError('Not enough samples to fit temporal line.')

        shuffle_idxs = []

        while True:
            batches = None
            for n_batch in range(self.batch_size):
                if len(shuffle_idxs) == 0:
                    shuffle_idxs = samples_idxs.copy()
                    np.random.shuffle (shuffle_idxs)

                idx = shuffle_idxs.pop()

                temporal_samples = []
                mult = np.random.randint(mult_max)+1
                for i in range( self.temporal_image_count ):
                    sample = samples[ idx+i*mult ]
                    try:
                        temporal_samples += SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)[0]
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(temporal_samples)) ]

                for i in range(len(temporal_samples)):
                    batches[i].append ( temporal_samples[i] )

            yield [ np.array(batch) for batch in batches]
