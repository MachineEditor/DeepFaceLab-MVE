import multiprocessing
import traceback

import cv2
import numpy as np

from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleHost, SampleProcessor,
                       SampleType)
from utils import iter_utils
from utils import mp_utils

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
                        **kwargs):

        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.add_sample_idx = add_sample_idx

        samples_host = SampleHost.mp_host (SampleType.FACE, self.samples_path)
        self.samples_len = len(samples_host.get_list())

        if self.samples_len == 0:
            raise ValueError('No training data provided.')

        index_host = mp_utils.IndexHost(self.samples_len)

        if random_ct_samples_path is not None:
            ct_samples_host = SampleHost.mp_host (SampleType.FACE, random_ct_samples_path)
            ct_index_host = mp_utils.IndexHost( len(ct_samples_host.get_list()) )
        else:
            ct_samples_host = None
            ct_index_host = None

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (samples_host.create_cli(), index_host.create_cli(), ct_samples_host.create_cli() if ct_index_host is not None else None, ct_index_host.create_cli() if ct_index_host is not None else None) )]
        else:
            self.generators_count = np.clip(multiprocessing.cpu_count(), 2, 4)
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (samples_host.create_cli(), index_host.create_cli(), ct_samples_host.create_cli() if ct_index_host is not None else None, ct_index_host.create_cli() if ct_index_host is not None else None), start_now=True ) for i in range(self.generators_count) ]

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
        samples, index_host, ct_samples, ct_index_host = param
        bs = self.batch_size
        while True:
            batches = None

            indexes = index_host.get(bs)
            ct_indexes = ct_index_host.get(bs) if ct_samples is not None else None

            for n_batch in range(bs):
                sample_idx = indexes[n_batch]
                sample = samples[ sample_idx ]
                ct_sample = ct_samples[ ct_indexes[n_batch] ] if ct_samples is not None else None

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
