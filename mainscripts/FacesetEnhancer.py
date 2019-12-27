import multiprocessing
import shutil

from DFLIMG import *
from interact import interact as io
from joblib import Subprocessor
from nnlib import nnlib
from utils import Path_utils
from utils.cv2_utils import *


class FacesetEnhancerSubprocessor(Subprocessor):
    
    #override
    def __init__(self, image_paths, output_dirpath, multi_gpu=False, cpu_only=False):
        self.image_paths = image_paths
        self.output_dirpath = output_dirpath
        self.result = []
        self.devices = FacesetEnhancerSubprocessor.get_devices_for_config(multi_gpu, cpu_only)
   
        super().__init__('FacesetEnhancer', FacesetEnhancerSubprocessor.Cli, 600)

    #override
    def on_clients_initialized(self):
        io.progress_bar (None, len (self.image_paths))
        
    #override
    def on_clients_finalized(self):
        io.progress_bar_close()
        
    #override
    def process_info_generator(self):
        base_dict = {'output_dirpath':self.output_dirpath}

        for (device_idx, device_type, device_name, device_total_vram_gb) in self.devices:
            client_dict = base_dict.copy()
            client_dict['device_idx'] = device_idx
            client_dict['device_name'] = device_name
            client_dict['device_type'] = device_type
            yield client_dict['device_name'], {}, client_dict

    #override
    def get_data(self, host_dict):        
        if len (self.image_paths) > 0:
            return self.image_paths.pop(0)
            
    #override
    def on_data_return (self, host_dict, data):
        self.image_paths.insert(0, data)
        
    #override
    def on_result (self, host_dict, data, result):
        io.progress_bar_inc(1)
        if result[0] == 1:
            self.result +=[ (result[1], result[2]) ]
            
    #override
    def get_result(self):
        return self.result
                   
    @staticmethod
    def get_devices_for_config (multi_gpu, cpu_only):
        backend = nnlib.device.backend
        if 'cpu' in backend:
            cpu_only = True

        if not cpu_only and backend == "plaidML":
            cpu_only = True

        if not cpu_only:
            devices = []
            if multi_gpu:
                devices = nnlib.device.getValidDevicesWithAtLeastTotalMemoryGB(2)

            if len(devices) == 0:
                idx = nnlib.device.getBestValidDeviceIdx()
                if idx != -1:
                    devices = [idx]

            if len(devices) == 0:
                cpu_only = True

            result = []
            for idx in devices:
                dev_name = nnlib.device.getDeviceName(idx)
                dev_vram = nnlib.device.getDeviceVRAMTotalGb(idx)

                result += [ (idx, 'GPU', dev_name, dev_vram) ]
                
            return result

        if cpu_only:
            return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in range( min(8, multiprocessing.cpu_count() // 2) ) ]
    
    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            device_idx   = client_dict['device_idx']
            cpu_only     = client_dict['device_type'] == 'CPU'
            self.output_dirpath = client_dict['output_dirpath']
            
            device_config = nnlib.DeviceConfig ( cpu_only=cpu_only, force_gpu_idx=device_idx, allow_growth=True)
            nnlib.import_all (device_config)
            
            device_vram = device_config.gpu_vram_gb[0]

            intro_str = 'Running on %s.' % (client_dict['device_name'])
            if not cpu_only and device_vram <= 2:
                intro_str += " Recommended to close all programs using this device."

            self.log_info (intro_str)

            from facelib import FaceEnhancer
            self.fe = FaceEnhancer()

        #override
        def process_data(self, filepath):
            try:
                dflimg = DFLIMG.load (filepath)
                if dflimg is None:
                    self.log_err ("%s is not a dfl image file" % (filepath.name) )
                else:
                    img = cv2_imread(filepath).astype(np.float32) / 255.0
                    
                    img = self.fe.enhance(img)
                    
                    img = np.clip (img*255, 0, 255).astype(np.uint8)
                    
                    output_filepath = self.output_dirpath / filepath.name
                    
                    cv2_imwrite ( str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )
                    dflimg.embed_and_set ( str(output_filepath) )
                    return (1, filepath, output_filepath)
            except:
                self.log_err (f"Exception occured while processing file {filepath}. Error: {traceback.format_exc()}")
        
            return (0, filepath, None)
            
def process_folder ( dirpath, multi_gpu=False, cpu_only=False ):
    output_dirpath = dirpath.parent / (dirpath.name + '_enhanced')
    output_dirpath.mkdir (exist_ok=True, parents=True)
            
    dirpath_parts = '/'.join( dirpath.parts[-2:])
    output_dirpath_parts = '/'.join( output_dirpath.parts[-2:] )
    io.log_info (f"Enhancing faceset in {dirpath_parts}")
    io.log_info ( f"Processing to {output_dirpath_parts}")

    output_images_paths = Path_utils.get_image_paths(output_dirpath)
    if len(output_images_paths) > 0:
        for filename in output_images_paths:
            Path(filename).unlink()
    
    image_paths = [Path(x) for x in Path_utils.get_image_paths( dirpath )]    
    result = FacesetEnhancerSubprocessor ( image_paths, output_dirpath, multi_gpu=multi_gpu, cpu_only=cpu_only).run()

    is_merge = io.input_bool (f"\r\nMerge {output_dirpath_parts} to {dirpath_parts} ? (y/n skip:y) : ", True)
    if is_merge:
        io.log_info (f"Copying processed files to {dirpath_parts}")
        
        for (filepath, output_filepath) in result:        
            shutil.copy (output_filepath, filepath)
            
        io.log_info (f"Removing {output_dirpath_parts}")
        shutil.rmtree(output_dirpath)
