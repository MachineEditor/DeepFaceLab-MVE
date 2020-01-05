import multiprocessing
import operator
import traceback
from pathlib import Path

import samplelib.PackedFaceset
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor
from interact import interact as io
from joblib import Subprocessor
from utils import Path_utils, mp_utils

from .Sample import Sample, SampleType


class SampleHost:




    samples_cache = dict()
    @staticmethod
    def get_person_id_max_count(samples_path):
        samples = None
        try:
            samples = samplelib.PackedFaceset.load(samples_path)
        except:
            io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}")

        if samples is None:
            raise ValueError("packed faceset not found.")
        persons_name_idxs = {}
        for sample in samples:
            persons_name_idxs[sample.person_name] = 0
        return len(list(persons_name_idxs.keys()))

    @staticmethod
    def host(sample_type, samples_path, number_of_clis):
        samples_cache = SampleHost.samples_cache

        if str(samples_path) not in samples_cache.keys():
            samples_cache[str(samples_path)] = [None]*SampleType.QTY

        samples = samples_cache[str(samples_path)]

        if            sample_type == SampleType.IMAGE:
            if  samples[sample_type] is None:
                samples[sample_type] = [ Sample(filename=filename) for filename in io.progress_bar_generator( Path_utils.get_image_paths(samples_path), "Loading") ]
        elif          sample_type == SampleType.FACE or \
                      sample_type == SampleType.FACE_TEMPORAL_SORTED:
            result = None

            if  samples[sample_type] is None:
                try:
                    result = samplelib.PackedFaceset.load(samples_path)
                except:
                    io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}")

                if result is not None:
                    io.log_info (f"Loaded {len(result)} packed faces from {samples_path}")

                if result is None:
                    result = SampleHost.load_face_samples( Path_utils.get_image_paths(samples_path) )

                if sample_type == SampleType.FACE_TEMPORAL_SORTED:
                    result = SampleHost.upgradeToFaceTemporalSortedSamples(result)

                samples[sample_type] = mp_utils.ListHost(result)

            list_host = samples[sample_type]

            clis = [ list_host.create_cli() for _ in range(number_of_clis) ]

            return clis

        return samples[sample_type]

    @staticmethod
    def load_face_samples ( image_paths):
        result = FaceSamplesLoaderSubprocessor(image_paths).run()
        sample_list = []

        for filename, \
                ( face_type,
                  shape,
                  landmarks,
                  ie_polys,
                  eyebrows_expand_mod,
                  source_filename,
                ) in result:
            sample_list.append( Sample(filename=filename,
                                        sample_type=SampleType.FACE,
                                        face_type=FaceType.fromString (face_type),
                                        shape=shape,
                                        landmarks=landmarks,
                                        ie_polys=ie_polys,
                                        eyebrows_expand_mod=eyebrows_expand_mod,
                                        source_filename=source_filename,
                                    ))
        return sample_list

    @staticmethod
    def upgradeToFaceTemporalSortedSamples( samples ):
        new_s = [ (s, s.source_filename) for s in samples]
        new_s = sorted(new_s, key=operator.itemgetter(1))

        return [ s[0] for s in new_s]


class FaceSamplesLoaderSubprocessor(Subprocessor):
    #override
    def __init__(self, image_paths ):
        self.image_paths = image_paths
        self.image_paths_len = len(image_paths)
        self.idxs = [*range(self.image_paths_len)]
        self.result = [None]*self.image_paths_len
        super().__init__('FaceSamplesLoader', FaceSamplesLoaderSubprocessor.Cli, 60, initialize_subprocesses_in_serial=False)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Loading", len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      }

    #override
    def get_data(self, host_dict):
        if len (self.idxs) > 0:
            idx = self.idxs.pop(0)
            return idx, self.image_paths[idx]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, dflimg = result
        self.result[idx] = (self.image_paths[idx], dflimg)
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            pass

        #override
        def process_data(self, data):
            idx, filename = data
            dflimg = DFLIMG.load (Path(filename))

            if dflimg is None:
                self.log_err (f"FaceSamplesLoader: {filename} is not a dfl image file.")
                data = None
            else:
                data = (dflimg.get_face_type(),
                        dflimg.get_shape(),
                        dflimg.get_landmarks(),
                        dflimg.get_ie_polys(),
                        dflimg.get_eyebrows_expand_mod(),
                        dflimg.get_source_filename() )

            return idx, data

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[1]
