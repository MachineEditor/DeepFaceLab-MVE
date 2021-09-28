import math
import multiprocessing
import traceback
from pathlib import Path

import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from merger import FrameInfo, InteractiveMergerSubprocessor, MergerConfig


def main (model_class_name=None,
          saved_models_path=None,
          training_data_src_path=None,
          force_model_name=None,
          input_path=None,
          output_path=None,
          output_mask_path=None,
          aligned_path=None,
          force_gpu_idxs=None,
          cpu_only=None):
    io.log_info ("Running merger.\r\n")

    try:
        if not input_path.exists():
            io.log_err('Input directory not found. Please ensure it exists.')
            return

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not output_mask_path.exists():
            output_mask_path.mkdir(parents=True, exist_ok=True)

        if not saved_models_path.exists():
            io.log_err('Model directory not found. Please ensure it exists.')
            return

        # Initialize model
        import models
        model = models.import_model(model_class_name)(is_training=False,
                                                      saved_models_path=saved_models_path,
                                                      force_gpu_idxs=force_gpu_idxs,
                                                      force_model_name=force_model_name,
                                                      cpu_only=cpu_only)

        predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()

        # Preparing MP functions
        predictor_func = MPFunc(predictor_func)

        run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
        xseg_256_extract_func = MPClassFuncOnDemand(XSegNet, 'extract',
                                                    name='XSeg',
                                                    resolution=256,
                                                    weights_file_root=saved_models_path,
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        face_enhancer_func = MPClassFuncOnDemand(FaceEnhancer, 'enhance',
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        is_interactive = io.input_bool ("Use interactive merger?", True) if not io.is_colab() else False

        if not is_interactive:
            cfg.ask_settings()
            
        subprocess_count = io.input_int("Number of workers?", max(8, multiprocessing.cpu_count()), 
                                        valid_range=[1, multiprocessing.cpu_count()], help_message="Specify the number of threads to process. A low value may affect performance. A high value may result in memory error. The value may not be greater than CPU cores." )

        input_path_image_paths = pathex.get_image_paths(input_path)

        if cfg.type == MergerConfig.TYPE_MASKED:
            if not aligned_path.exists():
                io.log_err('Aligned directory not found. Please ensure it exists.')
                return

            packed_samples = None
            try:
                packed_samples = samplelib.PackedFaceset.load(aligned_path)
            except:
                io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(aligned_path)}, {traceback.format_exc()}")


            if packed_samples is not None:
                io.log_info ("Using packed faceset.")
                def generator():
                    for sample in io.progress_bar_generator( packed_samples, "Collecting alignments"):
                        filepath = Path(sample.filename)
                        yield filepath, DFLIMG.load(filepath, loader_func=lambda x: sample.read_raw_file()  )
            else:
                def generator():
                    for filepath in io.progress_bar_generator( pathex.get_image_paths(aligned_path), "Collecting alignments"):
                        filepath = Path(filepath)
                        yield filepath, DFLIMG.load(filepath)

            alignments = {}
            multiple_faces_detected = False

            for filepath, dflimg in generator():
                if dflimg is None or not dflimg.has_data():
                    io.log_err (f"{filepath.name} is not a dfl image file")
                    continue

                source_filename = dflimg.get_source_filename()
                if source_filename is None:
                    continue

                source_filepath = Path(source_filename)
                source_filename_stem = source_filepath.stem

                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments_ar = alignments[ source_filename_stem ]
                alignments_ar.append ( (dflimg.get_source_landmarks(), filepath, source_filepath ) )

                if len(alignments_ar) > 1:
                    multiple_faces_detected = True

            if multiple_faces_detected:
                io.log_info ("")
                io.log_info ("Warning: multiple faces detected. Only one alignment file should refer one source file.")
                io.log_info ("")

            for a_key in list(alignments.keys()):
                a_ar = alignments[a_key]
                if len(a_ar) > 1:
                    for _, filepath, source_filepath in a_ar:
                        io.log_info (f"alignment {filepath.name} refers to {source_filepath.name} ")
                    io.log_info ("")

                alignments[a_key] = [ a[0] for a in a_ar]

            if multiple_faces_detected:
                io.log_info ("It is strongly recommended to process the faces separatelly.")
                io.log_info ("Use 'recover original filename' to determine the exact duplicates.")
                io.log_info ("")

            frames = [ InteractiveMergerSubprocessor.Frame( frame_info=FrameInfo(filepath=Path(p),
                                                                     landmarks_list=alignments.get(Path(p).stem, None)
                                                                    )
                                              )
                       for p in input_path_image_paths ]

            if multiple_faces_detected:
                io.log_info ("Warning: multiple faces detected. Motion blur will not be used.")
                io.log_info ("")
            else:
                s = 256
                local_pts = [ (s//2-1, s//2-1), (s//2-1,0) ] #center+up
                frames_len = len(frames)
                for i in io.progress_bar_generator( range(len(frames)) , "Computing motion vectors"):
                    fi_prev = frames[max(0, i-1)].frame_info
                    fi      = frames[i].frame_info
                    fi_next = frames[min(i+1, frames_len-1)].frame_info
                    if len(fi_prev.landmarks_list) == 0 or \
                       len(fi.landmarks_list) == 0 or \
                       len(fi_next.landmarks_list) == 0:
                            continue

                    mat_prev = LandmarksProcessor.get_transform_mat ( fi_prev.landmarks_list[0], s, face_type=FaceType.FULL)
                    mat      = LandmarksProcessor.get_transform_mat ( fi.landmarks_list[0]     , s, face_type=FaceType.FULL)
                    mat_next = LandmarksProcessor.get_transform_mat ( fi_next.landmarks_list[0], s, face_type=FaceType.FULL)

                    pts_prev = LandmarksProcessor.transform_points (local_pts, mat_prev, True)
                    pts      = LandmarksProcessor.transform_points (local_pts, mat, True)
                    pts_next = LandmarksProcessor.transform_points (local_pts, mat_next, True)

                    prev_vector = pts[0]-pts_prev[0]
                    next_vector = pts_next[0]-pts[0]

                    motion_vector = pts_next[0] - pts_prev[0]
                    fi.motion_power = npla.norm(motion_vector)

                    motion_vector = motion_vector / fi.motion_power if fi.motion_power != 0 else np.array([0,0],dtype=np.float32)

                    fi.motion_deg = -math.atan2(motion_vector[1],motion_vector[0])*180 / math.pi


        if len(frames) == 0:
            io.log_info ("No frames to merge in input_dir.")
        else:
            if False:
                pass
            else:
                InteractiveMergerSubprocessor (
                            is_interactive         = is_interactive,
                            merger_session_filepath = model.get_strpath_storage_for_file('merger_session.dat'),
                            predictor_func         = predictor_func,
                            predictor_input_shape  = predictor_input_shape,
                            face_enhancer_func     = face_enhancer_func,
                            xseg_256_extract_func = xseg_256_extract_func,
                            merger_config          = cfg,
                            frames                 = frames,
                            frames_root_path       = input_path,
                            output_path            = output_path,
                            output_mask_path       = output_mask_path,
                            model_iter             = model.get_iter(),
                            subprocess_count       = subprocess_count,
                        ).run()

        model.finalize()

    except Exception as e:
        print ( traceback.format_exc() )


"""
elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
filesdata = []
for filepath in io.progress_bar_generator(input_path_image_paths, "Collecting info"):
    filepath = Path(filepath)

    dflimg = DFLIMG.x(filepath)
    if dflimg is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        continue
    filesdata += [ ( FrameInfo(filepath=filepath, landmarks_list=[dflimg.get_landmarks()] ), dflimg.get_source_filename() ) ]

filesdata = sorted(filesdata, key=operator.itemgetter(1)) #sort by source_filename
frames = []
filesdata_len = len(filesdata)
for i in range(len(filesdata)):
    frame_info = filesdata[i][0]

    prev_temporal_frame_infos = []
    next_temporal_frame_infos = []

    for t in range (cfg.temporal_face_count):
        prev_frame_info = filesdata[ max(i -t, 0) ][0]
        next_frame_info = filesdata[ min(i +t, filesdata_len-1 )][0]

        prev_temporal_frame_infos.insert (0, prev_frame_info )
        next_temporal_frame_infos.append (   next_frame_info )

    frames.append ( InteractiveMergerSubprocessor.Frame(prev_temporal_frame_infos=prev_temporal_frame_infos,
                                                frame_info=frame_info,
                                                next_temporal_frame_infos=next_temporal_frame_infos) )
"""

#interpolate landmarks
#from facelib import LandmarksProcessor
#from facelib import FaceType
#a = sorted(alignments.keys())
#a_len = len(a)
#
#box_pts = 3
#box = np.ones(box_pts)/box_pts
#for i in range( a_len ):
#    if i >= box_pts and i <= a_len-box_pts-1:
#        af0 = alignments[ a[i] ][0] ##first face
#        m0 = LandmarksProcessor.get_transform_mat (af0, 256, face_type=FaceType.FULL)
#
#        points = []
#
#        for j in range(-box_pts, box_pts+1):
#            af = alignments[ a[i+j] ][0] ##first face
#            m = LandmarksProcessor.get_transform_mat (af, 256, face_type=FaceType.FULL)
#            p = LandmarksProcessor.transform_points (af, m)
#            points.append (p)
#
#        points = np.array(points)
#        points_len = len(points)
#        t_points = np.transpose(points, [1,0,2])
#
#        p1 = np.array ( [ int(np.convolve(x[:,0], box, mode='same')[points_len//2]) for x in t_points ] )
#        p2 = np.array ( [ int(np.convolve(x[:,1], box, mode='same')[points_len//2]) for x in t_points ] )
#
#        new_points = np.concatenate( [np.expand_dims(p1,-1),np.expand_dims(p2,-1)], -1 )
#
#        alignments[ a[i] ][0]  = LandmarksProcessor.transform_points (new_points, m0, True).astype(np.int32)
