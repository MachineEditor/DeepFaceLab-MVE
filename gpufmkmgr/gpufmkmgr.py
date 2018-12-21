import os
import sys
import contextlib

from utils import std_utils
from .pynvml import *


        
dlib_module = None
def import_dlib(device_idx, cpu_only=False):
    global dlib_module
    if dlib_module is not None:
        raise Exception ('Multiple import of dlib is not allowed, reorganize your program.')
        
    import dlib
    dlib_module = dlib
    if not cpu_only:
        dlib_module.cuda.set_device(device_idx)    
    return dlib_module

tf_module = None
tf_session = None
keras_module = None
keras_contrib_module = None
keras_vggface_module = None

def set_prefer_GPUConfig(gpu_config):
    global prefer_GPUConfig
    prefer_GPUConfig = gpu_config
    
def get_tf_session():
    global tf_session
    return tf_session

def import_tf( gpu_config = None ):
    global prefer_GPUConfig
    global tf_module
    global tf_session
    
    if gpu_config is None:
        gpu_config = prefer_GPUConfig
    else:
        prefer_GPUConfig = gpu_config
        
    if tf_module is not None:
        return tf_module

    if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
        suppressor = std_utils.suppress_stdout_stderr().__enter__()
    else:
        suppressor = None

    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        os.environ.pop('CUDA_VISIBLE_DEVICES')
    
    os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
    
    import tensorflow as tf
    tf_module = tf
    
    if gpu_config.cpu_only:
        config = tf_module.ConfigProto( device_count = {'GPU': 0} )
    else:    
        config = tf_module.ConfigProto()
        visible_device_list = ''
        for idx in gpu_config.gpu_idxs: visible_device_list += str(idx) + ','
        visible_device_list = visible_device_list[:-1]
        config.gpu_options.visible_device_list=visible_device_list
        config.gpu_options.force_gpu_compatible = True
        
    config.gpu_options.allow_growth = gpu_config.allow_growth
    
    tf_session = tf_module.Session(config=config)
        
    if suppressor is not None:  
        suppressor.__exit__()

    return tf_module

def finalize_tf():
    global tf_module
    global tf_session
    
    tf_session.close()
    tf_session = None
    tf_module = None

def get_keras():
    global keras_module
    return keras_module
    
def import_keras():
    global keras_module
    
    if keras_module is not None:
        return keras_module
        
    sess = get_tf_session()
    if sess is None:
        raise Exception ('No TF session found. Import TF first.')
        
    if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':
        suppressor = std_utils.suppress_stdout_stderr().__enter__()
        
    import keras     

    keras.backend.tensorflow_backend.set_session(sess)
    
    if 'TF_SUPPRESS_STD' in os.environ.keys() and os.environ['TF_SUPPRESS_STD'] == '1':        
        suppressor.__exit__()

    keras_module = keras
    return keras_module
    
def finalize_keras():
    global keras_module
    keras_module.backend.clear_session()
    keras_module = None
    
def import_keras_contrib():
    global keras_contrib_module
    
    if keras_contrib_module is not None:
        raise Exception ('Multiple import of keras_contrib is not allowed, reorganize your program.')
    import keras_contrib
    keras_contrib_module = keras_contrib
    return keras_contrib_module
    
def finalize_keras_contrib():
    global keras_contrib_module
    keras_contrib_module = None
    
def import_keras_vggface(optional=False):
    global keras_vggface_module
    
    if keras_vggface_module is not None:
        raise Exception ('Multiple import of keras_vggface_module is not allowed, reorganize your program.')

    try:
        import keras_vggface
    except:
        if optional:
            print ("Unable to import keras_vggface. It will not be used.")
        else:
            raise Exception ("Unable to import keras_vggface.")
        keras_vggface = None
        
    keras_vggface_module = keras_vggface
    return keras_vggface_module
    
def finalize_keras_vggface():
    global keras_vggface_module
    keras_vggface_module = None    

def hasNVML():
    try:
        nvmlInit()
        nvmlShutdown()
    except:
        return False
    return True    
 
#returns [ (device_idx, device_name), ... ]
def getDevicesWithAtLeastFreeMemory(freememsize):
    result = []
    
    nvmlInit()
    for i in range(0, nvmlDeviceGetCount() ):
        handle = nvmlDeviceGetHandleByIndex(i)
        memInfo = nvmlDeviceGetMemoryInfo( handle )
        if (memInfo.total - memInfo.used) >= freememsize:
            result.append (i)
        
    nvmlShutdown()
        
    return result
    
def getDevicesWithAtLeastTotalMemoryGB(totalmemsize_gb):
    result = []
    
    nvmlInit()
    for i in range(0, nvmlDeviceGetCount() ):
        handle = nvmlDeviceGetHandleByIndex(i)
        memInfo = nvmlDeviceGetMemoryInfo( handle )
        if (memInfo.total) >= totalmemsize_gb*1024*1024*1024:
            result.append (i)
        
    nvmlShutdown()
        
    return result
def getAllDevicesIdxsList ():
    nvmlInit()    
    result = [ i for i in range(0, nvmlDeviceGetCount() ) ]    
    nvmlShutdown()        
    return result
    
def getDeviceVRAMFree (idx):
    result = 0
    nvmlInit()
    if idx < nvmlDeviceGetCount():    
        handle = nvmlDeviceGetHandleByIndex(idx)
        memInfo = nvmlDeviceGetMemoryInfo( handle )
        result = (memInfo.total - memInfo.used)        
    nvmlShutdown()
    return result
    
def getDeviceVRAMTotalGb (idx):
    result = 0
    nvmlInit()
    if idx < nvmlDeviceGetCount():    
        handle = nvmlDeviceGetHandleByIndex(idx)
        memInfo = nvmlDeviceGetMemoryInfo( handle )
        result = memInfo.total / (1024*1024*1024)
    nvmlShutdown()
    return round(result)
    
def getBestDeviceIdx():
    nvmlInit()    
    idx = -1
    idx_mem = 0
    for i in range(0, nvmlDeviceGetCount() ):
        handle = nvmlDeviceGetHandleByIndex(i)
        memInfo = nvmlDeviceGetMemoryInfo( handle )
        if memInfo.total > idx_mem:
            idx = i
            idx_mem = memInfo.total

    nvmlShutdown()
    return idx
    
def getWorstDeviceIdx():
    nvmlInit()    
    idx = -1
    idx_mem = sys.maxsize
    for i in range(0, nvmlDeviceGetCount() ):
        handle = nvmlDeviceGetHandleByIndex(i)
        memInfo = nvmlDeviceGetMemoryInfo( handle )
        if memInfo.total < idx_mem:
            idx = i
            idx_mem = memInfo.total

    nvmlShutdown()
    return idx
    
def isValidDeviceIdx(idx):
    nvmlInit()    
    result = (idx < nvmlDeviceGetCount())
    nvmlShutdown()
    return result
    
def getDeviceIdxsEqualModel(idx):
    result = []
    
    nvmlInit()    
    idx_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()

    for i in range(0, nvmlDeviceGetCount() ):
        if nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() == idx_name:
            result.append (i)
         
    nvmlShutdown()
    return result
    
def getDeviceName (idx):
    result = ''
    nvmlInit()    
    if idx < nvmlDeviceGetCount():    
        result = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()
    nvmlShutdown()
    return result
    
    
class GPUConfig():    
    force_best_gpu_idx = -1
    multi_gpu = False
    force_gpu_idxs = None
    choose_worst_gpu = False
    gpu_idxs = []
    gpu_total_vram_gb = 0
    allow_growth = True
    cpu_only = False
    
    def __init__ (self, force_best_gpu_idx = -1, 
                        multi_gpu = False, 
                        force_gpu_idxs = None, 
                        choose_worst_gpu = False,
                        allow_growth = True,
                        cpu_only = False,
                        **in_options):
        if not hasNVML():
            cpu_only = True
            
        if cpu_only:
            self.cpu_only = cpu_only
        else:
            self.force_best_gpu_idx = force_best_gpu_idx
            self.multi_gpu = multi_gpu
            self.force_gpu_idxs = force_gpu_idxs
            self.choose_worst_gpu = choose_worst_gpu        
            self.allow_growth = allow_growth
      
            gpu_idx = force_best_gpu_idx if (force_best_gpu_idx >= 0 and isValidDeviceIdx(force_best_gpu_idx)) else getBestDeviceIdx() if not choose_worst_gpu else getWorstDeviceIdx()

            if force_gpu_idxs is not None:
                self.gpu_idxs = [ int(x) for x in force_gpu_idxs.split(',') ]
            else:
                if self.multi_gpu:
                    self.gpu_idxs = getDeviceIdxsEqualModel( gpu_idx )
                    if len(self.gpu_idxs) <= 1:
                        self.multi_gpu = False
                else:
                    self.gpu_idxs = [gpu_idx]
            
            self.gpu_total_vram_gb = getDeviceVRAMTotalGb ( self.gpu_idxs[0] )
        
prefer_GPUConfig = GPUConfig()