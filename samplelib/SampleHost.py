import multiprocessing
import operator
import traceback
from pathlib import Path
import pickle
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
    def load(sample_type, samples_path):
        samples_cache = SampleHost.samples_cache

        if str(samples_path) not in samples_cache.keys():
            samples_cache[str(samples_path)] = [None]*SampleType.QTY

        samples = samples_cache[str(samples_path)]

        if            sample_type == SampleType.IMAGE:
            if  samples[sample_type] is None:
                samples[sample_type] = [ Sample(filename=filename) for filename in io.progress_bar_generator( Path_utils.get_image_paths(samples_path), "Loading") ]
        
        elif          sample_type == SampleType.FACE:
            if  samples[sample_type] is None:
                try:
                    result = samplelib.PackedFaceset.load(samples_path)
                except:
                    io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}")

                if result is not None:
                    io.log_info (f"Loaded {len(result)} packed faces from {samples_path}")

                if result is None:
                    result = SampleHost.load_face_samples( Path_utils.get_image_paths(samples_path) )
                samples[sample_type] = result
                
        elif          sample_type == SampleType.FACE_TEMPORAL_SORTED:
                result = SampleHost.load (SampleType.FACE, samples_path)
                result = SampleHost.upgradeToFaceTemporalSortedSamples(result)
                samples[sample_type] = result
                
        return samples[sample_type]

    @staticmethod
    def load_face_samples ( image_paths):
        sample_list = []
        
        for filename in io.progress_bar_generator (image_paths, desc="Loading"):
            dflimg = DFLIMG.load (Path(filename))            
            if dflimg is None:
                io.log_err (f"{filename} is not a dfl image file.")
            else:                        
                sample_list.append( Sample(filename=filename,
                                           sample_type=SampleType.FACE,
                                           face_type=FaceType.fromString ( dflimg.get_face_type() ),
                                           shape=dflimg.get_shape(),
                                           landmarks=dflimg.get_landmarks(),
                                           ie_polys=dflimg.get_ie_polys(),
                                           eyebrows_expand_mod=dflimg.get_eyebrows_expand_mod(),
                                           source_filename=dflimg.get_source_filename(),
                                    ))
        return sample_list

    @staticmethod
    def upgradeToFaceTemporalSortedSamples( samples ):
        new_s = [ (s, s.source_filename) for s in samples]
        new_s = sorted(new_s, key=operator.itemgetter(1))

        return [ s[0] for s in new_s]
