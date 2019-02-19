import os
import sys
import argparse
from utils import Path_utils
from utils import os_utils
from pathlib import Path

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 2):
    raise Exception("This program requires at least Python 3.2")

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == "__main__":
    os_utils.set_process_lowest_prio()   
    parser = argparse.ArgumentParser()    
    parser.add_argument('--tf-suppress-std', action="store_true", dest="tf_suppress_std", default=False, help="Suppress tensorflow initialization info. May not works on some python builds such as anaconda python 3.6.4. If you can fix it, you are welcome.")
 
    subparsers = parser.add_subparsers()
    
    def process_extract(arguments):
        from mainscripts import Extractor        
        Extractor.main (
            input_dir=arguments.input_dir, 
            output_dir=arguments.output_dir, 
            debug=arguments.debug,
            face_type=arguments.face_type,
            detector=arguments.detector,
            multi_gpu=arguments.multi_gpu,
            cpu_only=arguments.cpu_only,
            manual_fix=arguments.manual_fix,
            manual_output_debug_fix=arguments.manual_output_debug_fix,
            manual_window_size=arguments.manual_window_size
            )
        
    extract_parser = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    extract_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    extract_parser.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    extract_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Writes debug images to [output_dir]_debug\ directory.")    
    extract_parser.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'head', 'avatar', 'mark_only'], default='full_face', help="Default 'full_face'. Don't change this option, currently all models uses 'full_face'")    
    extract_parser.add_argument('--detector', dest="detector", choices=['dlib','mt','manual'], default='dlib', help="Type of detector. Default 'dlib'. 'mt' (MTCNNv1) - faster, better, almost no jitter, perfect for gathering thousands faces for src-set. It is also good for dst-set, but can generate false faces in frames where main face not recognized! In this case for dst-set use either 'dlib' with '--manual-fix' or '--detector manual'. Manual detector suitable only for dst-set.")
    extract_parser.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    extract_parser.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    extract_parser.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
    extract_parser.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="Manual fix window size. Default: 1368.") 
    extract_parser.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU. Forces to use MT extractor.")    
    extract_parser.set_defaults (func=process_extract)
    
    def process_sort(arguments):        
        from mainscripts import Sorter
        Sorter.main (input_path=arguments.input_dir, sort_by_method=arguments.sort_by_method)
        
    sort_parser = subparsers.add_parser( "sort", help="Sort faces in a directory.")     
    sort_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    sort_parser.add_argument('--by', required=True, dest="sort_by_method", choices=("blur", "face", "face-dissim", "face-yaw", "face-pitch", "hist", "hist-dissim", "brightness", "hue", "black", "origname", "final", "test"), help="Method of sorting. 'origname' sort by original filename to recover original sequence." )
    sort_parser.set_defaults (func=process_sort)
    
    def process_util(arguments):        
        from mainscripts import Util
        
        if arguments.convert_png_to_jpg:
            Util.convert_png_to_jpg_folder (input_path=arguments.input_dir)
        
        if arguments.add_landmarks_debug_images:
            Util.add_landmarks_debug_images (input_path=arguments.input_dir)
        
    util_parser = subparsers.add_parser( "util", help="Utilities.")     
    util_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    util_parser.add_argument('--convert-png-to-jpg', action="store_true", dest="convert_png_to_jpg", default=False, help="Convert DeepFaceLAB PNG files to JPEG.")
    util_parser.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False, help="Add landmarks debug image for aligned faces.")
    util_parser.set_defaults (func=process_util)
    
    def process_train(arguments):      
        from mainscripts import Trainer
        Trainer.main (
            training_data_src_dir=arguments.training_data_src_dir, 
            training_data_dst_dir=arguments.training_data_dst_dir, 
            model_path=arguments.model_dir, 
            model_name=arguments.model_name,
            debug              = arguments.debug,
            #**options
            force_gpu_idx = arguments.force_gpu_idx,
            cpu_only      = arguments.cpu_only
            )
        
    train_parser = subparsers.add_parser( "train", help="Trainer") 
    train_parser.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of src-set.")
    train_parser.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of dst-set.")
    train_parser.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    train_parser.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    train_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")  
    train_parser.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
    train_parser.add_argument('--force-gpu-idx', type=int, dest="force_gpu_idx", default=-1, help="Force to choose this GPU idx.")

    train_parser.set_defaults (func=process_train)
    
    def process_convert(arguments):
        from mainscripts import Converter
        Converter.main (
            input_dir=arguments.input_dir, 
            output_dir=arguments.output_dir, 
            aligned_dir=arguments.aligned_dir,
            model_dir=arguments.model_dir, 
            model_name=arguments.model_name, 
            debug = arguments.debug,
            force_gpu_idx = arguments.force_gpu_idx,
            cpu_only = arguments.cpu_only
            )
        
    convert_parser = subparsers.add_parser( "convert", help="Converter") 
    convert_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    convert_parser.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the converted files will be stored.")
    convert_parser.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", help="Aligned directory. This is where the extracted of dst faces stored. Not used in AVATAR model.")
    convert_parser.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    convert_parser.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    convert_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug converter.")
    convert_parser.add_argument('--force-gpu-idx', type=int, dest="force_gpu_idx", default=-1, help="Force to choose this GPU idx.")
    convert_parser.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Convert on CPU.")
    
    convert_parser.set_defaults(func=process_convert)

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)
    
    arguments = parser.parse_args()
    if arguments.tf_suppress_std:
        os.environ['TF_SUPPRESS_STD'] = '1'

    #os.environ['force_plaidML'] = '1'
    
    arguments.func(arguments)

    print ("Done.")
    
'''
import code
code.interact(local=dict(globals(), **locals()))
'''