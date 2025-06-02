from pathlib import Path
from typing import Optional
import onnxruntime as ort
import numpy as np

from core import pathex
from core.leras import nn
from facelib import FaceType
from core.interact import interact as io

so = ort.SessionOptions()
so.log_severity_level = 3  # 3 = ERROR, 2 = WARNING, 1 = INFO, 0 = VERBOSE

# wrapper around onnx for merge
class Model(object):
    def __init__(self, is_training: bool, saved_models_path: str, force_gpu_idxs, force_model_name: Optional[str], cpu_only: bool, reduce_clutter = False, silent_start=False):
        if is_training:
            raise NotImplementedError('Training mode is not supported')


        path = self.get_model_path(saved_models_path, force_model_name=force_model_name)
        self.saved_models_path = Path(saved_models_path)
        if self.saved_models_path.exists() == False:
            raise ValueError("Model path is not found")

        self.init_base(force_gpu_idxs=force_gpu_idxs, cpu_only=cpu_only, silent_start=silent_start)

        if len(self.chosen_id) == 0:
            provider = "CPUExecutionProvider"
            self.session = ort.InferenceSession(path, providers=[provider])
        else: # GPU
            provider = "CUDAExecutionProvider"
            self.session = ort.InferenceSession(path, providers=[provider], provider_options=[{"device_id": self.chosen_id[0]}])
        inputs = self.session.get_inputs()
        if len(inputs) == 0:
            raise ValueError("Model has no inputs")

        if 'in_face' not in inputs[0].name:
            raise ValueError(f'Invalid model input name {inputs[0].name}')

        self.input_height, self.input_width = inputs[0].shape[1:3]

        if len(inputs) == 2:
            if 'morph_value' not in inputs[1].name:
                raise ValueError(f'Invalid model input name {inputs[1].name}')
            self.has_morph = True
        else:
            self.has_morph = False


    # model base init
    def init_base(self, silent_start = False, force_gpu_idxs = None, cpu_only = False):
        # Select GPU index
        if force_gpu_idxs:
            raise NotImplementedError('Force gpu idx not implemented')

        if silent_start:
            raise NotImplementedError('Silent start is not implemented')
            if force_gpu_idxs is not None:
                self.device_config = nn.DeviceConfig.GPUIndexes(force_gpu_idxs) if not cpu_only else nn.DeviceConfig.CPU()
                io.log_info (f"Silent start: choosed device{'s' if len(force_gpu_idxs) > 0 else ''} {'CPU' if self.device_config.cpu_only else [device.name for device in self.device_config.devices]}")
            else:
                self.device_config = nn.DeviceConfig.BestGPU()
                io.log_info (f"Silent start: choosed device {'CPU' if self.device_config.cpu_only else self.device_config.devices[0].name}")
        else:
            self.chosen_id = nn.ask_choose_device_idxs(choose_only_one = True, suggest_best_multi_gpu=True)
            self.device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or self.chosen_id) if not cpu_only else nn.DeviceConfig.CPU()

        nn.initialize(self.device_config)

        # Set face type
        input_face = io.input_str ("Face type", 'wf', ['h','mf','f','wf','head', 'custom'], help_message="Half / mid face / full face / whole face / head / custom. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()

        self.face_type = {'h'  : FaceType.HALF,
                    'mf' : FaceType.MID_FULL,
                    'f'  : FaceType.FULL,
                    'wf' : FaceType.WHOLE_FACE,
                    'custom' : FaceType.CUSTOM,
                    'head' : FaceType.HEAD}[ input_face ]


    # model base function
    def get_strpath_storage_for_file(self, filename):
        return str( self.saved_models_path / ( self.get_model_name() + '_' + filename) )

    def get_model_name(self):
        return 'dfm'

    def get_iter(self):
        return 0

    def get_model_path(self, model_folder: str, force_model_name: Optional[str])-> str:
        valid_models: list[str] = []

        for filepath in pathex.get_file_paths(model_folder):
            filepath_name = filepath.name

            if filepath_name.endswith('.dfm'):
                valid_models.append(str(filepath))

        if len(valid_models) == 0:
            raise FileNotFoundError('Model file not found')

        if force_model_name:
            valid_models = [model_file for model_file in valid_models if not model_file.startswith(force_model_name)]


        if len(valid_models) > 1:
            raise ValueError('Multiple model files found')


        return valid_models[0]


    def predictor_func (self, face, morph_value = None):
        face = nn.to_data_format(face[None,...], 'NHWC', "NHWC")

        if self.has_morph:
            out_face_mask, out_celeb, out_celeb_mask = self.session.run(None, {'in_face:0': face, 'morph_value:0':np.float32([morph_value]) })
        else:
            out_face_mask, out_celeb, out_celeb_mask = self.session.run(None, {'in_face:0': face})

        out_face_mask, out_celeb, out_celeb_mask = [ nn.to_data_format(x,"NCHW", 'NCHW').astype(np.float32) for x in [out_face_mask, out_celeb, out_celeb_mask] ]

        return out_celeb[0], out_celeb_mask[0][...,0], out_face_mask[0][...,0]


    #override
    # def get_MergerConfig(self):

    #     def predictor_morph(face, func_morph_factor=1.0):
    #         return self.predictor_func(face, func_morph_factor)

    #     import merger
    #     return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay', is_morphable=True)


    def get_MergerConfig(self):
        import merger

        if self.has_morph:
            def predictor_morph(face, func_morph_factor=1.0):
                return self.predictor_func(face, func_morph_factor)

            return predictor_morph, (self.input_width, self.input_height, 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay', is_morphable=True)
        else:
            return self.predictor_func, (self.input_width, self.input_height, 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')


    def finalize(self):
        self.session = None