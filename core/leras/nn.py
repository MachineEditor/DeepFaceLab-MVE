"""
Leras. 

like lighter keras.
This is my lightweight neural network library written from scratch
based on pure tensorflow without keras.

Provides:
+ full freedom of tensorflow operations without keras model's restrictions 
+ easy model operations like in PyTorch, but in graph mode (no eager execution)
+ convenient and understandable logic

Reasons why we cannot import tensorflow or any tensorflow.sub modules right here:
1) change env variables based on DeviceConfig before import tensorflow
2) multiprocesses will import tensorflow every spawn
"""

import os
import sys
from pathlib import Path
from core.interact import interact as io
from .device import Devices

class nn():
    current_DeviceConfig = None

    tf = None
    tf_sess = None
    tf_sess_config = None
    
    # Tensor ops
    tf_get_value = None
    tf_batch_set_value = None
    tf_gradients = None
    tf_average_gv_list = None
    tf_average_tensor_list = None
    tf_dot = None
    tf_gelu = None
    tf_upsample2d = None
    tf_upsample2d_bilinear = None
    tf_flatten = None
    tf_random_binomial = None
    tf_gaussian_blur = None
    tf_style_loss = None
    tf_dssim = None
    
    # Layers
    Saveable = None
    LayerBase = None
    ModelBase = None
    Conv2D = None
    Conv2DTranspose = None
    BlurPool = None
    Dense = None
    BatchNorm2D = None
    
    # Initializers
    initializers = None
    
    # Optimizers
    TFBaseOptimizer = None
    TFRMSpropOptimizer = None
    
    @staticmethod
    def initialize(device_config=None):
        if nn.tf is None:
            if device_config is None:
                device_config = nn.getCurrentDeviceConfig()
            else:
                nn.setCurrentDeviceConfig(device_config)

            if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
                os.environ.pop('CUDA_VISIBLE_DEVICES')

            os.environ['CUDA_​CACHE_​MAXSIZE'] = '536870912' #512Mb (32mb default)
            
            first_run = False
            
            if sys.platform[0:3] == 'win':
                devices_str = ""
                for device in device_config.devices:
                    devices_str += "_" + device.name.replace(' ','_')
                
                compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache' + devices_str)
                if not compute_cache_path.exists():
                    first_run = True
                os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)

            os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log errors only

            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)
  
            if first_run:
                io.log_info("Caching GPU kernels...")

            import tensorflow as tf            
            nn.tf = tf
            
            if device_config.cpu_only:
                config = tf.ConfigProto(device_count={'GPU': 0})
            else:                
                config = tf.ConfigProto()
                config.gpu_options.visible_device_list = ','.join([str(device.index) for device in device_config.devices])

            config.gpu_options.force_gpu_compatible = True
            config.gpu_options.allow_growth = True
            nn.tf_sess_config = config

            nn.tf_floatx = nn.tf.float32 #nn.tf.float16 if device_config.use_fp16 else nn.tf.float32
            nn.np_floatx = nn.tf_floatx.as_numpy_dtype
    
            from .tensor_ops import initialize_tensor_ops
            from .layers import initialize_layers
            from .initializers import initialize_initializers
            from .optimizers import initialize_optimizers
            
            initialize_tensor_ops(nn)
            initialize_layers(nn)
            initialize_initializers(nn)
            initialize_optimizers(nn)
            
        if nn.tf_sess is None:
            nn.tf_sess = tf.Session(config=nn.tf_sess_config)
            
    @staticmethod
    def initialize_main_env():
        Devices.initialize_main_env()
    
    @staticmethod
    def getCurrentDeviceConfig():
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        nn.current_DeviceConfig = device_config

    @staticmethod
    def tf_reset_session():
        if nn.tf is not None:
            if nn.tf_sess is not None:
                nn.tf.reset_default_graph()
                nn.tf_sess.close()
                nn.tf_sess = nn.tf.Session(config=nn.tf_sess_config)
            
    @staticmethod
    def tf_close_session():        
        if nn.tf_sess is not None:
            nn.tf.reset_default_graph()
            nn.tf_sess.close()
            nn.tf_sess = None

            
    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False, return_device_config=False):
        devices = Devices.getDevices()
        if len(devices) == 0:
            return []
        
        all_devices_indexes = [device.index for device in devices]
        
        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False
   
        if suggest_all_gpu:
            best_device_indexes = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device()) ]
        else:
            best_device_indexes = [ devices.get_best_device().index ]
        best_device_indexes = ",".join([str(x) for x in best_device_indexes])
                        
        io.log_info ("")
        if choose_only_one:
            io.log_info ("Choose one GPU idx.")
        else:
            io.log_info ("Choose one or several GPU idxs (separated by comma).")
        io.log_info ("")
        
        if allow_cpu:
            io.log_info ("[CPU] : CPU")
        for device in devices:
            io.log_info (f"  [{device.index}] : {device.name}")
        
        io.log_info ("")
        
        while True:
            try:
                if choose_only_one:
                    choosed_idxs = io.input_str("Which GPU index to choose?", best_device_indexes)
                else:
                    choosed_idxs = io.input_str("Which GPU indexes to choose?", best_device_indexes)
                
                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break
                
                choosed_idxs = [ int(x) for x in choosed_idxs.split(',') ]
                
                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break                    
                else:
                    if all( [idx in all_devices_indexes for idx in choosed_idxs] ):
                        break
            except:
                pass
        io.log_info ("")
        
        if return_device_config:
            return nn.DeviceConfig.GPUIndexes(choosed_idxs)
        else:        
            return choosed_idxs

    class DeviceConfig():    
        def __init__ (self, devices=None):
            devices = devices or []       
            
            if not isinstance(devices, Devices):
                devices = Devices(devices)
                 
            self.devices = devices                   
            self.cpu_only = len(devices) == 0      
            
        @staticmethod
        def BestGPU():            
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()
            
            return nn.DeviceConfig([devices.get_best_device()])
            
        @staticmethod
        def WorstGPU():     
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()
                    
            return nn.DeviceConfig([devices.get_worst_device()])
            
        @staticmethod
        def GPUIndexes(indexes):
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []
                
            return nn.DeviceConfig(devices)
            
        @staticmethod
        def CPU():            
            return nn.DeviceConfig([])
