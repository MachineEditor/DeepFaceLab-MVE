import sys
import ctypes
import os
import json
import numpy as np

#you can set DFL_TF_MIN_REQ_CAP manually for your build
#the reason why we cannot check tensorflow.version is it requires import tensorflow
tf_min_req_cap = int(os.environ.get("DFL_TF_MIN_REQ_CAP", 35))

class device:
    backend = None
    class Config():
        force_gpu_idx = -1
        multi_gpu = False
        force_gpu_idxs = None
        choose_worst_gpu = False
        gpu_idxs = []
        gpu_names = []
        gpu_compute_caps = []
        gpu_vram_gb = []
        allow_growth = True
        use_fp16 = False
        cpu_only = False
        backend = None
        def __init__ (self, force_gpu_idx = -1,
                            multi_gpu = False,
                            force_gpu_idxs = None,
                            choose_worst_gpu = False,
                            allow_growth = True,
                            use_fp16 = False,
                            cpu_only = False,
                            **in_options):

            self.backend = device.backend
            self.use_fp16 = use_fp16
            self.cpu_only = cpu_only

            if not self.cpu_only:
                self.cpu_only = (self.backend == "tensorflow-cpu")

            if not self.cpu_only:
                self.force_gpu_idx = force_gpu_idx
                self.multi_gpu = multi_gpu
                self.force_gpu_idxs = force_gpu_idxs
                self.choose_worst_gpu = choose_worst_gpu
                self.allow_growth = allow_growth

                self.gpu_idxs = []

                if force_gpu_idxs is not None:
                    for idx in force_gpu_idxs.split(','):
                        idx = int(idx)
                        if device.isValidDeviceIdx(idx):
                            self.gpu_idxs.append(idx)
                else:
                    gpu_idx = force_gpu_idx if (force_gpu_idx >= 0 and device.isValidDeviceIdx(force_gpu_idx)) else device.getBestValidDeviceIdx() if not choose_worst_gpu else device.getWorstValidDeviceIdx()
                    if gpu_idx != -1:
                        if self.multi_gpu:
                            self.gpu_idxs = device.getDeviceIdxsEqualModel( gpu_idx )
                            if len(self.gpu_idxs) <= 1:
                                self.multi_gpu = False
                        else:
                            self.gpu_idxs = [gpu_idx]

                self.cpu_only = (len(self.gpu_idxs) == 0)


            if not self.cpu_only:
                self.gpu_names = []
                self.gpu_compute_caps = []
                self.gpu_vram_gb = []
                for gpu_idx in self.gpu_idxs:
                    self.gpu_names += [device.getDeviceName(gpu_idx)]
                    self.gpu_compute_caps += [ device.getDeviceComputeCapability(gpu_idx) ]
                    self.gpu_vram_gb += [ device.getDeviceVRAMTotalGb(gpu_idx) ]
                self.cpu_only = (len(self.gpu_idxs) == 0)
            else:
                self.gpu_names = ['CPU']
                self.gpu_compute_caps = [99]
                self.gpu_vram_gb = [0]

            if self.cpu_only:
                self.backend = "tensorflow-cpu"

    @staticmethod
    def getValidDeviceIdxsEnumerator():
        if device.backend == "plaidML":
            for i in range(plaidML_devices_count):
                yield i
        elif device.backend == "tensorflow":
            for dev in cuda_devices:
                yield dev['index']

    @staticmethod
    def getValidDevicesWithAtLeastTotalMemoryGB(totalmemsize_gb):
        result = []
        if device.backend == "plaidML":
            for i in device.getValidDeviceIdxsEnumerator():
                if plaidML_devices[i]['globalMemSize'] >= totalmemsize_gb*1024*1024*1024:
                     result.append (i)
        elif device.backend == "tensorflow":
            for dev in cuda_devices:
                if dev['total_mem'] >= totalmemsize_gb*1024*1024*1024:
                    result.append ( dev['index'] )

        return result

    @staticmethod
    def getValidDevicesIdxsWithNamesList():
        if device.backend == "plaidML":
            return [ (i, plaidML_devices[i]['description'] ) for i in device.getValidDeviceIdxsEnumerator() ]
        elif device.backend == "tensorflow":
            return [ ( dev['index'], dev['name'] ) for dev in cuda_devices ]
        elif device.backend == "tensorflow-cpu":
            return [ (0, 'CPU') ]

    @staticmethod
    def getDeviceVRAMTotalGb (idx):
        if device.backend == "plaidML":
            if idx < plaidML_devices_count:
                return plaidML_devices[idx]['globalMemSize'] / (1024*1024*1024)
        elif device.backend == "tensorflow":
            for dev in cuda_devices:
                if idx == dev['index']:
                    return round ( dev['total_mem'] / (1024*1024*1024) )
            return 0

    @staticmethod
    def getBestValidDeviceIdx():
        if device.backend == "plaidML":
            idx = -1
            idx_mem = 0
            for i in device.getValidDeviceIdxsEnumerator():
                total = plaidML_devices[i]['globalMemSize']
                if total > idx_mem:
                    idx = i
                    idx_mem = total

            return idx
        elif device.backend == "tensorflow":
            idx = -1
            idx_mem = 0
            for dev in cuda_devices:
                if dev['total_mem'] > idx_mem:
                    idx = dev['index']
                    idx_mem = dev['total_mem']

            return idx

    @staticmethod
    def getWorstValidDeviceIdx():
        if device.backend == "plaidML":
            idx = -1
            idx_mem = sys.maxsize
            for i in device.getValidDeviceIdxsEnumerator():
                total = plaidML_devices[i]['globalMemSize']
                if total < idx_mem:
                    idx = i
                    idx_mem = total

            return idx
        elif device.backend == "tensorflow":
            idx = -1
            idx_mem = sys.maxsize
            for dev in cuda_devices:
                if dev['total_mem'] < idx_mem:
                    idx = dev['index']
                    idx_mem = dev['total_mem']

            return idx

    @staticmethod
    def isValidDeviceIdx(idx):
        if device.backend == "plaidML":
            return idx in [*device.getValidDeviceIdxsEnumerator()]
        elif device.backend == "tensorflow":
            for dev in cuda_devices:
                if idx == dev['index']:
                    return True
        return False

    @staticmethod
    def getDeviceIdxsEqualModel(idx):
        if device.backend == "plaidML":
            result = []
            idx_name = plaidML_devices[idx]['description']
            for i in device.getValidDeviceIdxsEnumerator():
                if plaidML_devices[i]['description'] == idx_name:
                    result.append (i)

            return result
        elif device.backend == "tensorflow":
            result = []
            idx_name = device.getDeviceName(idx)
            for dev in cuda_devices:
                if dev['name'] == idx_name:
                    result.append ( dev['index'] )
                    

            return result

    @staticmethod
    def getDeviceName (idx):
        if device.backend == "plaidML":
            if idx < plaidML_devices_count:
                return plaidML_devices[idx]['description']
        elif device.backend == "tensorflow":
            for dev in cuda_devices:
                if dev['index'] == idx:
                    return dev['name']

        return None

    @staticmethod
    def getDeviceID (idx):
        if device.backend == "plaidML":
            if idx < plaidML_devices_count:
                return plaidML_devices[idx]['id'].decode()

        return None

    @staticmethod
    def getDeviceComputeCapability(idx):
        if device.backend == "plaidML":
            return 99
        elif device.backend == "tensorflow":
            for dev in cuda_devices:
                if dev['index'] == idx:
                    return dev['cc']
        return 0

plaidML_build = os.environ.get("DFL_PLAIDML_BUILD", "0") == "1"
plaidML_devices = None
plaidML_devices_count = 0
cuda_devices = None

if plaidML_build:
    if plaidML_devices is None:
        plaidML_devices = []
        # Using plaidML OpenCL backend to determine system devices
        try:
            os.environ['PLAIDML_EXPERIMENTAL'] = 'false' #this enables work plaidML without run 'plaidml-setup'
            import plaidml
            ctx = plaidml.Context()
            for d in plaidml.devices(ctx, return_all=True)[0]:
                details = json.loads(d.details)
                if details['type'] == 'CPU': #skipping opencl-CPU
                    continue
                plaidML_devices += [ {'id':d.id,
                                    'globalMemSize' : int(details['globalMemSize']),
                                    'description' : d.description.decode()
                                }]
            ctx.shutdown()
        except:
            pass
    plaidML_devices_count = len(plaidML_devices)
    if plaidML_devices_count != 0:
        device.backend = "plaidML"
else:      
    if cuda_devices is None:
        cuda_devices = []
        libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll')
        cuda = None
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except:
                continue
            else:
                break
        
        if cuda is not None:
            nGpus = ctypes.c_int()
            name = b' ' * 200
            cc_major = ctypes.c_int()
            cc_minor = ctypes.c_int()
            freeMem = ctypes.c_size_t()
            totalMem = ctypes.c_size_t()

            result = ctypes.c_int()
            device_t = ctypes.c_int()
            context = ctypes.c_void_p()
            error_str = ctypes.c_char_p()

            if cuda.cuInit(0) == 0 and \
                cuda.cuDeviceGetCount(ctypes.byref(nGpus)) == 0:
                for i in range(nGpus.value):
                    if cuda.cuDeviceGet(ctypes.byref(device_t), i) != 0 or \
                        cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device_t) != 0 or \
                        cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device_t) != 0:
                        continue

                    if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device_t) == 0:
                        if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                            cc = cc_major.value * 10 + cc_minor.value
                            if cc >= tf_min_req_cap:
                                cuda_devices.append ( {'index':i,
                                                       'name':name.split(b'\0', 1)[0].decode(),                                               
                                                       'total_mem':totalMem.value,
                                                       'free_mem':freeMem.value,
                                                       'cc':cc
                                                      }
                                                    )
                        cuda.cuCtxDetach(context)    
        
    if len(cuda_devices) != 0:
        device.backend = "tensorflow"

if device.backend is None:
    device.backend = "tensorflow-cpu"
