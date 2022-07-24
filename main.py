if __name__ == "__main__":
    # Uncomment to start DFL with PDB
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    # Fix for linux
    import multiprocessing
    multiprocessing.set_start_method("spawn")

    from core.leras import nn
    nn.initialize_main_env()
    import os
    import sys
    import time
    import argparse

    from core import pathex
    from core import osex
    from pathlib import Path
    from core.interact import interact as io

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception("This program requires at least Python 3.6")

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    exit_code = 0

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def process_extract(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.main( detector                = arguments.detector,
                        extract_from_video      = arguments.extract_from_video,
                        input_video             = Path(arguments.input_video) if arguments.input_video is not None else None,
                        chunk_size              = arguments.chunk_size,
                        input_path              = Path(arguments.input_dir),
                        output_path             = Path(arguments.output_dir),
                        output_debug            = arguments.output_debug,
                        manual_fix              = arguments.manual_fix,
                        manual_output_debug_fix = arguments.manual_output_debug_fix,
                        manual_window_size      = arguments.manual_window_size,
                        face_type               = arguments.face_type,
                        max_faces_from_image    = arguments.max_faces_from_image,
                        image_size              = arguments.image_size,
                        jpeg_quality            = arguments.jpeg_quality,
                        fps                     = arguments.fps,
                        cpu_only                = arguments.cpu_only,
                        force_gpu_idxs          = [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                      )

    p = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    p.add_argument('--detector', dest="detector", choices=['s3fd','manual'], default=None, help="Type of detector.")
    p.add_argument('--extract-from-video', dest='extract_from_video', action="store_true", default=False, help='Extract aligned images directly from video file')
    p.add_argument('--input-video', required=False, action=fixPathAction, dest="input_video", help="Input video to be processed. Specify .*-extension to find first file.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    p.add_argument('--output-debug', action="store_true", dest="output_debug", default=None, help="Writes debug images to <output-dir>_debug\ directory.")
    p.add_argument('--no-output-debug', action="store_false", dest="output_debug", default=None, help="Don't writes debug images to <output-dir>_debug\ directory.")
    p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'whole_face', 'head', 'mark_only'], default=None)
    p.add_argument('--max-faces-from-image', type=int, dest="max_faces_from_image", default=None, help="Max faces from image.")
    p.add_argument('--image-size', type=int, dest="image_size", default=None, help="Output image size.")
    p.add_argument('--jpeg-quality', type=int, dest="jpeg_quality", default=None, help="Jpeg quality.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="How many frames of every second of the video will be extracted. 0 - full fps.")
    p.add_argument('--chunk-size', type=int, dest='chunk_size', default=None, help='When extract from video is enabled allows to choose the maximum number of frames that DFL can save in memory')
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="Manual fix window size. Default: 1368.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU..")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")

    p.set_defaults (func=process_extract)

    def process_sort(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Sorter
        Sorter.main (input_path=Path(arguments.input_dir), sort_by_method=arguments.sort_by_method)

    p = subparsers.add_parser( "sort", help="Sort faces in a directory.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--by', dest="sort_by_method", default=None, choices=("blur", "motion-blur", "face-yaw", "face-pitch", "face-source-rect-size", "hist", "hist-dissim", "brightness", "hue", "black", "origname", "oneface", "final", "final-fast", "absdiff"), help="Method of sorting. 'origname' sort by original filename to recover original sequence." )
    p.set_defaults (func=process_sort)

    def process_util(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Util

        if arguments.add_landmarks_debug_images:
            Util.add_landmarks_debug_images (input_path=arguments.input_dir)

        if arguments.recover_original_aligned_filename:
            Util.recover_original_aligned_filename (input_path=arguments.input_dir)

        if arguments.save_faceset_metadata:
            Util.save_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.restore_faceset_metadata:
            Util.restore_faceset_metadata_folder (input_path=arguments.input_dir)

        if arguments.pack_faceset:
            io.log_info ("Performing faceset packing...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.pack( Path(arguments.input_dir), ext=arguments.archive_type)

        if arguments.unpack_faceset:
            io.log_info ("Performing faceset unpacking...\r\n")
            from samplelib import PackedFaceset
            PackedFaceset.unpack( Path(arguments.input_dir) )

        if arguments.export_faceset_mask:
            io.log_info ("Exporting faceset mask..\r\n")
            Util.export_faceset_mask( Path(arguments.input_dir) )
            
    p = subparsers.add_parser( "util", help="Utilities.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False, help="Add landmarks debug image for aligned faces.")
    p.add_argument('--recover-original-aligned-filename', action="store_true", dest="recover_original_aligned_filename", default=False, help="Recover original aligned filename.")
    p.add_argument('--save-faceset-metadata', action="store_true", dest="save_faceset_metadata", default=False, help="Save faceset metadata to file.")
    p.add_argument('--restore-faceset-metadata', action="store_true", dest="restore_faceset_metadata", default=False, help="Restore faceset metadata to file. Image filenames must be the same as used with save.")
    p.add_argument('--pack-faceset', action="store_true", dest="pack_faceset", default=False, help="")
    p.add_argument('--unpack-faceset', action="store_true", dest="unpack_faceset", default=False, help="")
    p.add_argument('--export-faceset-mask', action="store_true", dest="export_faceset_mask", default=False, help="")
    p.add_argument('--archive-type', dest="archive_type", choices=['zip', 'pak'], default=None)

    p.set_defaults (func=process_util)

    def process_train(arguments):
        osex.set_process_lowest_prio()


        kwargs = {'model_class_name'         : arguments.model_name,
                  'saved_models_path'        : Path(arguments.model_dir),
                  'training_data_src_path'   : Path(arguments.training_data_src_dir),
                  'training_data_dst_path'   : Path(arguments.training_data_dst_dir),
                  'pretraining_data_path'    : Path(arguments.pretraining_data_dir) if arguments.pretraining_data_dir is not None else None,
                  'pretrained_model_path'    : Path(arguments.pretrained_model_dir) if arguments.pretrained_model_dir is not None else None,
                  'src_pak_name'             : arguments.src_pak_name,
                  'dst_pak_name'             : arguments.dst_pak_name,
                  'no_preview'               : arguments.no_preview,
                  'force_model_name'         : arguments.force_model_name,
                  'force_gpu_idxs'           : [ int(x) for x in arguments.force_gpu_idxs.split(',') ] if arguments.force_gpu_idxs is not None else None,
                  'cpu_only'                 : arguments.cpu_only,
                  'silent_start'             : arguments.silent_start,
                  'execute_programs'         : [ [int(x[0]), x[1] ] for x in arguments.execute_program ],
                  'debug'                    : arguments.debug,
                  'saving_time'              : arguments.saving_time,
                  'tensorboard_dir'          : arguments.tensorboard_dir,
                  'start_tensorboard'        : arguments.start_tensorboard,
                  'flask_preview'            : arguments.flask_preview,
                  'config_training_file'     : arguments.config_training_file,
                  'auto_gen_config'          : arguments.auto_gen_config,
                  'gen_snapshot'             : arguments.gen_snapshot,
                  'reduce_clutter'           : arguments.reduce_clutter
                  }
        from mainscripts import Trainer
        Trainer.main(**kwargs)

    p = subparsers.add_parser( "train", help="Trainer")
    p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of extracted SRC faceset.")
    p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of extracted DST faceset.")
    p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None, help="Optional dir of extracted faceset that will be used in pretraining mode.")
    p.add_argument('--src-pak-name', required=False, dest='src_pak_name', type=str, default=None, help='Name of the src faceset pack to use')
    p.add_argument('--dst-pak-name', required=False, dest='dst_pak_name', type=str, default=None, help='Name of the dst faceset pack to use')
    p.add_argument('--pretrained-model-dir', action=fixPathAction, dest="pretrained_model_dir", default=None, help="Optional dir of pretrain model files. (Currently only for Quick96).")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Saved models dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")
    p.add_argument('--saving-time', type=int, dest="saving_time", default=25, help="Model saving interval.")
    p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False, help="Disable preview window.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    p.add_argument('--silent-start', action="store_true", dest="silent_start", default=False, help="Silent start. Automatically chooses Best GPU and last used model.")
    p.add_argument('--tensorboard-logdir', action=fixPathAction, dest="tensorboard_dir", help="Directory of the tensorboard output files")
    p.add_argument('--start-tensorboard', action="store_true", dest="start_tensorboard", default=False, help="Automatically start the tensorboard server preconfigured to the tensorboard-logdir")
    p.add_argument('--config-training-file', action=fixPathAction, dest="config_training_file", help="Path to custom yaml configuration file")
    p.add_argument('--auto-gen-config', action="store_true", dest="auto_gen_config", default=False, help="Saves a configuration file for each model used in the trainer. It'll have the same model name")
    p.add_argument('--reduce-clutter', action="store_true", dest="reduce_clutter", default=False, help='Remove options that are not used from printed summary')
    
    p.add_argument('--gen-snapshot', action="store_true", dest="gen_snapshot", default=False, help="Generate a set snapshot only.")
    p.add_argument('--flask-preview', action="store_true", dest="flask_preview", default=False,
                   help="Launches a flask server to view the previews in a web browser")

    p.add_argument('--execute-program', dest="execute_program", default=[], action='append', nargs='+')
    p.set_defaults (func=process_train)
    
    def process_exportdfm(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import ExportDFM
        ExportDFM.main(model_class_name = arguments.model_name, saved_models_path = Path(arguments.model_dir))

    p = subparsers.add_parser( "exportdfm", help="Export model to use in DeepFaceLive.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Saved models dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.set_defaults (func=process_exportdfm)

    def process_ampconverter(arguments):
        from mainscripts import AmpConverter
        AmpConverter.main(saved_models_path = Path(arguments.model_dir))

    p = subparsers.add_parser( "ampconverter", help="Rename model files in order to be used with AMPModel. Only for AMP model.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Saved models dir.")
    p.set_defaults (func=process_ampconverter)
    
    def process_merge(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import Merger
        Merger.main ( model_class_name       = arguments.model_name,
                      saved_models_path      = Path(arguments.model_dir),
                      force_model_name       = arguments.force_model_name,
                      input_path             = Path(arguments.input_dir),
                      output_path            = Path(arguments.output_dir),
                      output_mask_path       = Path(arguments.output_mask_dir),
                      aligned_path           = Path(arguments.aligned_dir) if arguments.aligned_dir is not None else None,
                      pak_name               = arguments.pak_name,
                      force_gpu_idxs         = arguments.force_gpu_idxs,
                      cpu_only               = arguments.cpu_only)

    p = subparsers.add_parser( "merge", help="Merger")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the merged files will be stored.")
    p.add_argument('--output-mask-dir', required=True, action=fixPathAction, dest="output_mask_dir", help="Output mask directory. This is where the mask files will be stored.")
    p.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", default=None, help="Aligned directory. This is where the extracted of dst faces stored.")
    p.add_argument('--pak-name', required=False, dest='pak_name', type=str, default=None, help='Name of the faceset pack to use')
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=pathex.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Model class name.")
    p.add_argument('--force-model-name', dest="force_model_name", default=None, help="Forcing to choose model name from model/ folder.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Merge on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")
    p.add_argument('--reduce-clutter', action="store_true", dest="reduce_clutter", default=False, help='Remove options that are not used from printed summary')
    p.set_defaults(func=process_merge)

    videoed_parser = subparsers.add_parser( "videoed", help="Video processing.").add_subparsers()

    def process_videoed_extract_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.extract_video (arguments.input_file, arguments.output_dir, arguments.output_ext, arguments.fps)
    p = videoed_parser.add_parser( "extract-video", help="Extract images from video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted images will be stored.")
    p.add_argument('--output-ext', dest="output_ext", default=None, help="Image format (extension) of output files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="How many frames of every second of the video will be extracted. 0 - full fps.")
    p.set_defaults(func=process_videoed_extract_video)

    def process_videoed_cut_video(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.cut_video (arguments.input_file,
                           arguments.from_time,
                           arguments.to_time,
                           arguments.audio_track_id,
                           arguments.bitrate)
    p = videoed_parser.add_parser( "cut-video", help="Cut video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--from-time', dest="from_time", default=None, help="From time, for example 00:00:00.000")
    p.add_argument('--to-time', dest="to_time", default=None, help="To time, for example 00:00:00.000")
    p.add_argument('--audio-track-id', type=int, dest="audio_track_id", default=None, help="Specify audio track id.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.set_defaults(func=process_videoed_cut_video)

    def process_videoed_denoise_image_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.denoise_image_sequence (arguments.input_dir, arguments.factor)
    p = videoed_parser.add_parser( "denoise-image-sequence", help="Denoise sequence of images, keeping sharp edges. Helps to remove pixel shake from the predicted face.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory to be processed.")
    p.add_argument('--factor', type=int, dest="factor", default=None, help="Denoise factor (1-20).")
    p.set_defaults(func=process_videoed_denoise_image_sequence)

    def process_videoed_video_from_sequence(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.video_from_sequence (input_dir      = arguments.input_dir,
                                     output_file    = arguments.output_file,
                                     reference_file = arguments.reference_file,
                                     ext      = arguments.ext,
                                     fps      = arguments.fps,
                                     bitrate  = arguments.bitrate,
                                     include_audio = arguments.include_audio,
                                     lossless = arguments.lossless)

    p = videoed_parser.add_parser( "video-from-sequence", help="Make video from image sequence.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-file', required=True, action=fixPathAction, dest="output_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--reference-file', action=fixPathAction, dest="reference_file", help="Reference file used to determine proper FPS and transfer audio from it. Specify .*-extension to find first file.")
    p.add_argument('--ext', dest="ext", default='png', help="Image format (extension) of input files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="FPS of output file. Overwritten by reference-file.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.add_argument('--include-audio', action="store_true", dest="include_audio", default=False, help="Include audio from reference file.")
    p.add_argument('--lossless', action="store_true", dest="lossless", default=False, help="PNG codec.")

    p.set_defaults(func=process_videoed_video_from_sequence)

    facesettool_parser = subparsers.add_parser( "facesettool", help="Faceset tools.").add_subparsers()

    def process_faceset_enhancer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetEnhancer
        FacesetEnhancer.process_folder ( Path(arguments.input_dir),
                                         cpu_only=arguments.cpu_only,
                                         force_gpu_idxs=arguments.force_gpu_idxs
                                       )

    p = facesettool_parser.add_parser ("enhance", help="Enhance details in DFL faceset.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Process on CPU.")
    p.add_argument('--force-gpu-idxs', dest="force_gpu_idxs", default=None, help="Force to choose GPU indexes separated by comma.")

    p.set_defaults(func=process_faceset_enhancer)
    
    
    p = facesettool_parser.add_parser ("resize", help="Resize DFL faceset.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")

    def process_faceset_resizer(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import FacesetResizer
        FacesetResizer.process_folder ( Path(arguments.input_dir) )
    p.set_defaults(func=process_faceset_resizer)
    
    def process_dev_test(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import dev_misc
        dev_misc.dev_test( arguments.input_dir )

    p = subparsers.add_parser( "dev_test", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_dev_test)

    # ========== XSeg
    xseg_parser = subparsers.add_parser( "xseg", help="XSeg tools.").add_subparsers()

    p = xseg_parser.add_parser( "editor", help="XSeg editor.")

    def process_xsegeditor(arguments):
        osex.set_process_lowest_prio()
        from XSegEditor import XSegEditor
        global exit_code
        exit_code = XSegEditor.start (Path(arguments.input_dir))

    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")

    p.set_defaults (func=process_xsegeditor)

    p = xseg_parser.add_parser( "apply", help="Apply trained XSeg model to the extracted faces.")

    def process_xsegapply(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.apply_xseg (Path(arguments.input_dir), Path(arguments.model_dir))
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir")
    p.set_defaults (func=process_xsegapply)


    p = xseg_parser.add_parser( "remove", help="Remove applied XSeg masks from the extracted faces.")
    def process_xsegremove(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.remove_xseg (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegremove)


    p = xseg_parser.add_parser( "remove_labels", help="Remove XSeg labels from the extracted faces.")
    def process_xsegremovelabels(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.remove_xseg_labels (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegremovelabels)


    p = xseg_parser.add_parser( "fetch", help="Copies faces containing XSeg polygons in <input_dir>_xseg dir.")

    def process_xsegfetch(arguments):
        osex.set_process_lowest_prio()
        from mainscripts import XSegUtil
        XSegUtil.fetch_xseg (Path(arguments.input_dir) )
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir")
    p.set_defaults (func=process_xsegfetch)

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    from utils.logo import print_community_info
    print_community_info()

    arguments = parser.parse_args()
    arguments.func(arguments)

    if exit_code == 0:
        print ("Done.")

    exit(exit_code)

'''
import code
code.interact(local=dict(globals(), **locals()))
'''
