import operator
import traceback
from pathlib import Path

from facelib import FaceType, LandmarksProcessor
from interact import interact as io
from utils import Path_utils, mp_utils
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG

from .Sample import Sample, SampleType

import samplelib.PackedFaceset

class SampleHost:
    samples_cache = dict()
    host_cache = dict()

    @staticmethod
    def get_person_id_max_count(samples_path):
        return len ( Path_utils.get_all_dir_names(samples_path) )

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
                result = None
                try:
                    result = samplelib.PackedFaceset.load(samples_path)  
                except:
                    io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(samples_dat_path)}, {traceback.format_exc()}")
                    
                if result is not None:
                    io.log_info (f"Loaded {len(result)} packed samples from {samples_path}")
                            
                if result is None:
                    result = SampleHost.load_face_samples( Path_utils.get_image_paths(samples_path) )

                samples[sample_type] = result

        elif          sample_type == SampleType.FACE_TEMPORAL_SORTED:
            if  samples[sample_type] is None:
                samples[sample_type] = SampleHost.upgradeToFaceTemporalSortedSamples( SampleHost.load(SampleType.FACE, samples_path) )

        return samples[sample_type]

    @staticmethod
    def mp_host(sample_type, samples_path):
        result = SampleHost.load (sample_type, samples_path)

        host_cache = SampleHost.host_cache
        if str(samples_path) not in host_cache.keys():
            host_cache[str(samples_path)] = [None]*SampleType.QTY
        hosts = host_cache[str(samples_path)]

        if hosts[sample_type] is None:
            hosts[sample_type] = mp_utils.ListHost(result)

        return hosts[sample_type]

    @staticmethod
    def load_face_samples ( image_paths, silent=False):
        sample_list = []

        for filename in (image_paths if silent else io.progress_bar_generator( image_paths, "Loading")):
            filename_path = Path(filename)
            try:
                if filename_path.suffix == '.png':
                    dflimg = DFLPNG.load ( str(filename_path) )
                elif filename_path.suffix == '.jpg':
                    dflimg = DFLJPG.load ( str(filename_path) )
                else:
                    dflimg = None

                if dflimg is None:
                    io.log_err ("load_face_samples: %s is not a dfl image file required for training" % (filename_path.name) )
                    continue

                landmarks = dflimg.get_landmarks()
                pitch_yaw_roll = dflimg.get_pitch_yaw_roll()
                eyebrows_expand_mod = dflimg.get_eyebrows_expand_mod()

                if pitch_yaw_roll is None:
                    pitch_yaw_roll = LandmarksProcessor.estimate_pitch_yaw_roll(landmarks)

                sample_list.append( Sample(filename=filename,
                                           sample_type=SampleType.FACE,
                                           face_type=FaceType.fromString (dflimg.get_face_type()),
                                           shape=dflimg.get_shape(),
                                           landmarks=landmarks,
                                           ie_polys=dflimg.get_ie_polys(),
                                           pitch_yaw_roll=pitch_yaw_roll,
                                           eyebrows_expand_mod=eyebrows_expand_mod,
                                           source_filename=dflimg.get_source_filename(),
                                           fanseg_mask_exist=dflimg.get_fanseg_mask() is not None, ) )
            except:
                io.log_err ("Unable to load %s , error: %s" % (filename, traceback.format_exc() ) )

        return sample_list

    @staticmethod
    def upgradeToFaceTemporalSortedSamples( samples ):
        new_s = [ (s, s.source_filename) for s in samples]
        new_s = sorted(new_s, key=operator.itemgetter(1))

        return [ s[0] for s in new_s]
