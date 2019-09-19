from . import Extractor
from . import Sorter
from pathlib import Path
from utils import Path_utils
import shutil
from interact import interact as io

def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_out')
    
    dir_names = Path_utils.get_all_dir_names(input_path)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        
    
    
    for dir_name in dir_names:
        
        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        io.log_info (f"Processing: {str(cur_input_path)} ")
        
        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)

        Extractor.main( str(cur_input_path),
              str(cur_output_path),
              detector='s3fd',
              image_size=256,
              face_type='full_face',
              max_faces_from_image=1,
              device_args=device_args )
              
        io.log_info (f"Sorting: {str(cur_input_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')
        
        try:
            io.log_info (f"Removing: {str(cur_input_path)} ")
            shutil.rmtree(cur_input_path)
        except:
            io.log_info (f"unable to remove: {str(cur_input_path)} ")