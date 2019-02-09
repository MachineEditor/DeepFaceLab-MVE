from .pynvml import *

try:
    nvmlInit()
    hasNVML = True
except:
    hasNVML = False

class devicelib:
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
        
        def __init__ (self, force_gpu_idx = -1, 
                            multi_gpu = False, 
                            force_gpu_idxs = None, 
                            choose_worst_gpu = False,
                            allow_growth = True,
                            use_fp16 = False,
                            cpu_only = False,
                            **in_options):

            self.use_fp16 = use_fp16
            if cpu_only:
                self.cpu_only = True
            else:
                self.force_gpu_idx = force_gpu_idx
                self.multi_gpu = multi_gpu
                self.force_gpu_idxs = force_gpu_idxs
                self.choose_worst_gpu = choose_worst_gpu        
                self.allow_growth = allow_growth
          
                self.gpu_idxs = []

                if force_gpu_idxs is not None:
                    for idx in force_gpu_idxs.split(','):
                        idx = int(idx)
                        if devicelib.isValidDeviceIdx(idx):
                            self.gpu_idxs.append(idx)     
                else:
                    gpu_idx = force_gpu_idx if (force_gpu_idx >= 0 and devicelib.isValidDeviceIdx(force_gpu_idx)) else devicelib.getBestDeviceIdx() if not choose_worst_gpu else devicelib.getWorstDeviceIdx()
                    if gpu_idx != -1:
                        if self.multi_gpu:
                            self.gpu_idxs = devicelib.getDeviceIdxsEqualModel( gpu_idx )
                            if len(self.gpu_idxs) <= 1:
                                self.multi_gpu = False
                        else:
                            self.gpu_idxs = [gpu_idx]
                            
                self.cpu_only = (len(self.gpu_idxs) == 0)
 
                if not self.cpu_only:
                    self.gpu_names = []
                    self.gpu_compute_caps = []
                    for gpu_idx in self.gpu_idxs:
                        self.gpu_names += [devicelib.getDeviceName(gpu_idx)]
                        self.gpu_compute_caps += [ devicelib.getDeviceComputeCapability ( gpu_idx ) ]
                        self.gpu_vram_gb += [ devicelib.getDeviceVRAMTotalGb ( gpu_idx ) ]
                        
    @staticmethod
    def getDevicesWithAtLeastTotalMemoryGB(totalmemsize_gb):
        if not hasNVML:
            return [0]
            
        result = []
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            memInfo = nvmlDeviceGetMemoryInfo( handle )
            if (memInfo.total) >= totalmemsize_gb*1024*1024*1024:
                result.append (i)
        return result
        
    @staticmethod
    def getAllDevicesIdxsList():
        if not hasNVML:
            return [0]
            
        return [ i for i in range(0, nvmlDeviceGetCount() ) ]
        
    @staticmethod
    def getAllDevicesIdxsWithNamesList():
        if not hasNVML:
            return [ (0, devicelib.getDeviceName(0) ) ]
  
        return [ (i, nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() ) for i in range(nvmlDeviceGetCount() ) ]
        
    @staticmethod
    def getDeviceVRAMFree (idx):
        if not hasNVML:
            return 2

        if idx < nvmlDeviceGetCount():    
            memInfo = nvmlDeviceGetMemoryInfo( nvmlDeviceGetHandleByIndex(idx) )
            return memInfo.total - memInfo.used

        return 0
        
    @staticmethod
    def getDeviceVRAMTotalGb (idx):
        if not hasNVML:
            return 2
            
        if idx < nvmlDeviceGetCount():    
            memInfo = nvmlDeviceGetMemoryInfo(  nvmlDeviceGetHandleByIndex(idx) )
            return round ( memInfo.total / (1024*1024*1024) )

        return 0
        
    @staticmethod
    def getBestDeviceIdx():
        if not hasNVML:
            return 0

        idx = -1
        idx_mem = 0
        for i in range( nvmlDeviceGetCount() ):
            memInfo = nvmlDeviceGetMemoryInfo( nvmlDeviceGetHandleByIndex(i) )
            if memInfo.total > idx_mem:
                idx = i
                idx_mem = memInfo.total

        return idx
        
    @staticmethod
    def getWorstDeviceIdx():
        if not hasNVML:
            return 0

        idx = -1
        idx_mem = sys.maxsize
        for i in range( nvmlDeviceGetCount() ):
            memInfo = nvmlDeviceGetMemoryInfo( nvmlDeviceGetHandleByIndex(i) )
            if memInfo.total < idx_mem:
                idx = i
                idx_mem = memInfo.total

        return idx
        
    @staticmethod
    def isValidDeviceIdx(idx):
        if not hasNVML:
            return (idx == 0)
   
        return (idx < nvmlDeviceGetCount())
        
    @staticmethod
    def getDeviceIdxsEqualModel(idx):
        if not hasNVML:
            return [0] if idx == 0 else []            
        
        result = []  
        idx_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()
        for i in range( nvmlDeviceGetCount() ):
            if nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() == idx_name:
                result.append (i)

        return result
        
    @staticmethod
    def getDeviceName (idx):
        if not hasNVML:
            return 'Generic GeForce GPU'
            
        if idx < nvmlDeviceGetCount():    
            return nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()

        return None
        
    @staticmethod
    def getDeviceComputeCapability(idx):
        if not hasNVML:
            return 99 if idx == 0 else 0
            
        result = 0  
        if idx < nvmlDeviceGetCount():    
            result = nvmlDeviceGetCudaComputeCapability(nvmlDeviceGetHandleByIndex(idx))
        return result[0] * 10 + result[1]
