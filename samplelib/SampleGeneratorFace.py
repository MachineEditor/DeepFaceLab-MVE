import multiprocessing
import pickle
import time
import traceback

import cv2
import numpy as np

from core import mplib
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)


'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path, debug=False, batch_size=1,
                        random_ct_samples_path=None,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        add_sample_idx=False,
                        generators_count=4,
                        **kwargs):

        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.add_sample_idx = add_sample_idx

        if self.debug:
            self.generators_count = 1
        else:
            self.generators_count = max(1, generators_count)

        samples = SampleLoader.load (SampleType.FACE, self.samples_path)
        self.samples_len = len(samples)

        if self.samples_len == 0:
            raise ValueError('No training data provided.')

        index_host = mplib.IndexHost(self.samples_len)

        if random_ct_samples_path is not None:
            ct_samples = SampleLoader.load (SampleType.FACE, random_ct_samples_path)
            ct_index_host = mplib.IndexHost( len(ct_samples) )
        else:
            ct_samples = None
            ct_index_host = None

        pickled_samples = pickle.dumps(samples, 4)
        ct_pickled_samples = pickle.dumps(ct_samples, 4) if ct_samples is not None else None

        if self.debug:
            self.generators = [ThisThreadGenerator ( self.batch_func, (pickled_samples, index_host.create_cli(), ct_pickled_samples, ct_index_host.create_cli() if ct_index_host is not None else None) )]
        else:
            self.generators = [SubprocessGenerator ( self.batch_func, (pickled_samples, index_host.create_cli(), ct_pickled_samples, ct_index_host.create_cli() if ct_index_host is not None else None), start_now=False ) \
                               for i in range(self.generators_count) ]
                               
            SubprocessGenerator.start_in_parallel( self.generators )

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        pickled_samples, index_host, ct_pickled_samples, ct_index_host = param

        samples = pickle.loads(pickled_samples)
        ct_samples = pickle.loads(ct_pickled_samples) if ct_pickled_samples is not None else None

        bs = self.batch_size
        while True:
            batches = None

            indexes = index_host.multi_get(bs)
            ct_indexes = ct_index_host.multi_get(bs) if ct_samples is not None else None

            t = time.time()
            for n_batch in range(bs):
                sample_idx = indexes[n_batch]
                sample = samples[sample_idx]

                ct_sample = None
                if ct_samples is not None:
                    ct_sample = ct_samples[ct_indexes[n_batch]]

                try:
                    x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                except:
                    raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]
                    if self.add_sample_idx:
                        batches += [ [] ]
                        i_sample_idx = len(batches)-1

                for i in range(len(x)):
                    batches[i].append ( x[i] )

                if self.add_sample_idx:
                    batches[i_sample_idx].append (sample_idx)

            yield [ np.array(batch) for batch in batches]
