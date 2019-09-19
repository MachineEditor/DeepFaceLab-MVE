import math
import multiprocessing
import operator
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import numpy.linalg as npla

import imagelib
from converters import (ConverterConfig, ConvertFaceAvatar, ConvertMasked,
                        FrameInfo)
from facelib import FaceType, FANSegmentator, LandmarksProcessor
from interact import interact as io
from joblib import SubprocessFunctionCaller, Subprocessor
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG

from .ConverterScreen import Screen, ScreenManager

CONVERTER_DEBUG = False

class ConvertSubprocessor(Subprocessor):

    class Frame(object):
        def __init__(self, prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None):
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filename = None

            self.idx = None
            self.cfg = None
            self.is_done = False
            self.is_processing = False
            self.is_shown = False
            self.image = None

    class ProcessingFrame(object):
        def __init__(self, idx=None,
                           cfg=None,
                           prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None,
                           output_filename=None,
                           need_return_image = False):
            self.idx = idx
            self.cfg = cfg
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filename = output_filename

            self.need_return_image = need_return_image
            if self.need_return_image:
                self.image = None

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.log_info ('Running on %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.predictor_func = client_dict['predictor_func']
            self.predictor_input_shape = client_dict['predictor_input_shape']
            self.superres_func = client_dict['superres_func']

            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            from nnlib import nnlib
            #model process ate all GPU mem,
            #so we cannot use GPU for any TF operations in converter processes
            #therefore forcing active_DeviceConfig to CPU only
            nnlib.active_DeviceConfig = nnlib.DeviceConfig (cpu_only=True)

            def blursharpen_func (img, sharpen_mode=0, kernel_size=3, amount=100):
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if amount > 0:
                    if sharpen_mode == 1: #box
                        kernel = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
                        kernel[ kernel_size//2, kernel_size//2] = 1.0
                        box_filter = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
                        kernel = kernel + (kernel - box_filter) * amount
                        return cv2.filter2D(img, -1, kernel)
                    elif sharpen_mode == 2: #gaussian
                        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size) , 0)
                        img = cv2.addWeighted(img, 1.0 + (0.5 * amount), blur, -(0.5 * amount), 0)  
                        return img
                elif amount < 0:
                    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size) , 0)
                    img = cv2.addWeighted(img, 1.0 - a / 50.0, blur, a /50.0, 0)  
                    return img                    
                return img
            self.blursharpen_func = blursharpen_func

            self.fanseg_by_face_type = {}
            self.fanseg_input_size = 256

            def fanseg_extract(face_type, *args, **kwargs):
                fanseg = self.fanseg_by_face_type.get(face_type, None)
                if self.fanseg_by_face_type.get(face_type, None) is None:
                    fanseg = FANSegmentator( self.fanseg_input_size , FaceType.toString( face_type ) )
                    self.fanseg_by_face_type[face_type] = fanseg

                return fanseg.extract(*args, **kwargs)

            self.fanseg_extract_func = fanseg_extract
            
            import ebsynth
            def ebs_ct(*args, **kwargs):                    
                return ebsynth.color_transfer(*args, **kwargs)
                
            self.ebs_ct_func = ebs_ct
            
            return None

        #override
        def process_data(self, pf): #pf=ProcessingFrame
            cfg = pf.cfg.copy()
            cfg.blursharpen_func = self.blursharpen_func
            cfg.superres_func = self.superres_func
            cfg.ebs_ct_func = self.ebs_ct_func

            frame_info = pf.frame_info

            filename = frame_info.filename
            landmarks_list = frame_info.landmarks_list

            filename_path = Path(filename)
            output_filename = pf.output_filename
            need_return_image = pf.need_return_image

            if len(landmarks_list) == 0:
                self.log_info ( 'no faces found for %s, copying without faces' % (filename_path.name) )

                if filename_path.suffix == '.png':
                    shutil.copy (filename, output_filename )
                else:
                    img_bgr = cv2_imread(filename)
                    cv2_imwrite (output_filename, img_bgr)

                if need_return_image:
                    img_bgr = cv2_imread(filename)
                    pf.image = img_bgr
            else:
                if cfg.type == ConverterConfig.TYPE_MASKED:
                    cfg.fanseg_input_size = self.fanseg_input_size
                    cfg.fanseg_extract_func = self.fanseg_extract_func

                    try:
                        final_img = ConvertMasked (self.predictor_func, self.predictor_input_shape, cfg, frame_info)
                    except Exception as e:
                        e_str = traceback.format_exc()
                        if 'MemoryError' in e_str:
                            raise Subprocessor.SilenceException
                        else:
                            raise Exception( 'Error while converting file [%s]: %s' % (filename, e_str) )

                elif cfg.type == ConverterConfig.TYPE_FACE_AVATAR:
                    final_img = ConvertFaceAvatar (self.predictor_func, self.predictor_input_shape, 
                                                   cfg, pf.prev_temporal_frame_infos,
                                                        pf.frame_info,
                                                        pf.next_temporal_frame_infos )

                if output_filename is not None and final_img is not None:
                    cv2_imwrite (output_filename, final_img )

                if need_return_image:
                    pf.image = final_img

            return pf

        #overridable
        def get_data_name (self, pf):
            #return string identificator of your data
            return pf.frame_info.filename

    #override
    def __init__(self, is_interactive, converter_session_filepath, predictor_func, predictor_input_shape, converter_config, frames, output_path, model_iter):
        if len (frames) == 0:
            raise ValueError ("len (frames) == 0")

        super().__init__('Converter', ConvertSubprocessor.Cli, 86400 if CONVERTER_DEBUG else 60, io_loop_sleep_time=0.001, initialize_subprocesses_in_serial=False)

        self.is_interactive = is_interactive
        self.converter_session_filepath = Path(converter_session_filepath)
        self.converter_config = converter_config

        #dummy predict and sleep, tensorflow caching kernels. If remove it, sometime conversion speed can be x2 slower
        predictor_func (dummy_predict=True)
        time.sleep(2)

        self.predictor_func_host, self.predictor_func = SubprocessFunctionCaller.make_pair(predictor_func)
        self.predictor_input_shape = predictor_input_shape

        self.dcscn = None
        self.ranksrgan = None
        def superres_func(mode, *args, **kwargs):
            if mode == 1:
                if self.ranksrgan is None:
                    self.ranksrgan = imagelib.RankSRGAN()
                return self.ranksrgan.upscale(*args, **kwargs)

        self.dcscn_host, self.superres_func = SubprocessFunctionCaller.make_pair(superres_func)

        self.output_path = output_path
        self.model_iter = model_iter

        self.prefetch_frame_count = self.process_count = min(6,multiprocessing.cpu_count())

        session_data = None
        if self.is_interactive and self.converter_session_filepath.exists():
            
            if io.input_bool ("Use saved session? (y/n skip:y) : ", True):
                try:
                    with open( str(self.converter_session_filepath), "rb") as f:
                        session_data = pickle.loads(f.read())
                except Exception as e:
                    pass

        self.frames = frames
        self.frames_idxs = [ *range(len(self.frames)) ]
        self.frames_done_idxs = []

        if self.is_interactive and session_data is not None:
            s_frames = session_data.get('frames', None)
            s_frames_idxs = session_data.get('frames_idxs', None)
            s_frames_done_idxs = session_data.get('frames_done_idxs', None)
            s_model_iter = session_data.get('model_iter', None)

            frames_equal = (s_frames is not None) and \
                           (s_frames_idxs is not None) and \
                           (s_frames_done_idxs is not None) and \
                           (s_model_iter is not None) and \
                           (len(frames) == len(s_frames))

            if frames_equal:
                for i in range(len(frames)):
                    frame = frames[i]
                    s_frame = s_frames[i]
                    if frame.frame_info.filename != s_frame.frame_info.filename:
                        frames_equal = False
                    if not frames_equal:
                        break

            if frames_equal:
                io.log_info ('Using saved session from ' + '/'.join (self.converter_session_filepath.parts[-2:]) )
                self.frames = s_frames
                self.frames_idxs = s_frames_idxs
                self.frames_done_idxs = s_frames_done_idxs

                if self.model_iter != s_model_iter:
                    #model is more trained, recompute all frames
                    for frame in self.frames:
                        frame.is_done = False

                if self.model_iter != s_model_iter or \
                    len(self.frames_idxs) == 0:
                    #rewind to begin if model is more trained or all frames are done
                   
                    while len(self.frames_done_idxs) > 0:
                        prev_frame = self.frames[self.frames_done_idxs.pop()]
                        self.frames_idxs.insert(0, prev_frame.idx)

                if len(self.frames_idxs) != 0:
                    cur_frame = self.frames[self.frames_idxs[0]]
                    cur_frame.is_shown = False

            if not frames_equal:
                session_data = None

        if session_data is None:
            for filename in Path_utils.get_image_paths(self.output_path): #remove all images in output_path
                Path(filename).unlink()

            frames[0].cfg = self.converter_config.copy()

        for i in range( len(self.frames) ):
            frame = self.frames[i]
            frame.idx = i
            frame.output_filename = self.output_path / ( Path(frame.frame_info.filename).stem + '.png' )



    #override
    def process_info_generator(self):
        r = [0] if CONVERTER_DEBUG else range(self.process_count)

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'predictor_func': self.predictor_func,
                                      'predictor_input_shape' : self.predictor_input_shape,
                                      'superres_func': self.superres_func,
                                      'stdin_fd': sys.stdin.fileno() if CONVERTER_DEBUG else None
                                      }

    #overridable optional
    def on_clients_initialized(self):
        io.progress_bar ("Converting", len (self.frames_idxs), initial=len(self.frames_done_idxs) )

        self.process_remain_frames = not self.is_interactive
        self.is_interactive_quitting = not self.is_interactive

        if self.is_interactive:
            help_images = {
                    ConverterConfig.TYPE_MASKED :      cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_converter_masked.jpg') ),
                    ConverterConfig.TYPE_FACE_AVATAR : cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_converter_face_avatar.jpg') ),
                }

            self.main_screen = Screen(initial_scale_to_width=1368, image=None, waiting_icon=True)
            self.help_screen = Screen(initial_scale_to_height=768, image=help_images[self.converter_config.type], waiting_icon=False)
            self.screen_manager = ScreenManager( "Converter", [self.main_screen, self.help_screen], capture_keys=True )
            self.screen_manager.set_current (self.help_screen)
            self.screen_manager.show_current()

    #overridable optional
    def on_clients_finalized(self):
        io.progress_bar_close()

        if self.is_interactive:
            self.screen_manager.finalize()

            for frame in self.frames:
                frame.output_filename = None
                frame.image = None

            session_data = {
                'frames': self.frames,
                'frames_idxs': self.frames_idxs,
                'frames_done_idxs': self.frames_done_idxs,
                'model_iter' : self.model_iter,
            }
            self.converter_session_filepath.write_bytes( pickle.dumps(session_data) )

            io.log_info ("Session is saved to " + '/'.join (self.converter_session_filepath.parts[-2:]) )

    cfg_change_keys = ['`','1', '2', '3', '4', '5', '6', '7', '8', '9',
                                 'q', 'a', 'w', 's', 'e', 'd', 'r', 'f', 't', 'g','y','h','u','j',
                                 'z', 'x', 'c', 'v', 'b','n'   ]
    #override
    def on_tick(self):
        self.predictor_func_host.process_messages()
        self.dcscn_host.process_messages()

        go_prev_frame = False
        go_prev_frame_overriding_cfg = False
        go_next_frame = self.process_remain_frames
        go_next_frame_overriding_cfg = False

        cur_frame = None
        if len(self.frames_idxs) != 0:
            cur_frame = self.frames[self.frames_idxs[0]]

        if self.is_interactive:
            self.main_screen.set_waiting_icon(False)

            if not self.is_interactive_quitting and not self.process_remain_frames:
                if cur_frame is not None:
                    if not cur_frame.is_shown:
                        if cur_frame.is_done:
                            cur_frame.is_shown = True
                            io.log_info (cur_frame.cfg.to_string( cur_frame.frame_info.filename_short) )

                            if cur_frame.image is None:
                                cur_frame.image = cv2_imread ( cur_frame.output_filename)
                                if cur_frame.image is None:
                                    cur_frame.is_done = False #unable to read? recompute then
                                    cur_frame.is_shown = False
                            self.main_screen.set_image(cur_frame.image)
                        else:
                            self.main_screen.set_waiting_icon(True)

                else:
                    self.main_screen.set_image(None)
            else:
                self.main_screen.set_image(None)
                self.main_screen.set_waiting_icon(True)

            self.screen_manager.show_current()

            key_events = self.screen_manager.get_key_events()
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == 9: #tab
                self.screen_manager.switch_screens()
            else:
                if key == 27: #esc
                    self.is_interactive_quitting = True
                elif self.screen_manager.get_current() is self.main_screen:
                    if chr_key in self.cfg_change_keys:
                        self.process_remain_frames = False

                        if cur_frame is not None:
                            cfg = cur_frame.cfg
                            prev_cfg = cfg.copy()

                            if cfg.type == ConverterConfig.TYPE_MASKED:
                                if chr_key == '`':
                                    cfg.set_mode(0)
                                elif key >= ord('1') and key <= ord('9'):
                                    cfg.set_mode( key - ord('0') )
                                elif chr_key == 'q':
                                    cfg.add_hist_match_threshold(1 if not shift_pressed else 5)
                                elif chr_key == 'a':
                                    cfg.add_hist_match_threshold(-1 if not shift_pressed else -5)
                                elif chr_key == 'w':
                                    cfg.add_erode_mask_modifier(1 if not shift_pressed else 5)
                                elif chr_key == 's':
                                    cfg.add_erode_mask_modifier(-1 if not shift_pressed else -5)
                                elif chr_key == 'e':
                                    cfg.add_blur_mask_modifier(1 if not shift_pressed else 5)
                                elif chr_key == 'd':
                                    cfg.add_blur_mask_modifier(-1 if not shift_pressed else -5)
                                elif chr_key == 'r':
                                    cfg.add_motion_blur_power(1 if not shift_pressed else 5)
                                elif chr_key == 'f':
                                    cfg.add_motion_blur_power(-1 if not shift_pressed else -5)
                                elif chr_key == 't':
                                    cfg.add_color_degrade_power(1 if not shift_pressed else 5)
                                elif chr_key == 'g':
                                    cfg.add_color_degrade_power(-1 if not shift_pressed else -5)
                                elif chr_key == 'y':
                                    cfg.add_blursharpen_amount(1 if not shift_pressed else 5)
                                elif chr_key == 'h':
                                    cfg.add_blursharpen_amount(-1 if not shift_pressed else -5)
                                elif chr_key == 'u':
                                    cfg.add_output_face_scale(1 if not shift_pressed else 5)
                                elif chr_key == 'j':
                                    cfg.add_output_face_scale(-1 if not shift_pressed else -5)

                                elif chr_key == 'z':
                                    cfg.toggle_masked_hist_match()
                                elif chr_key == 'x':
                                    cfg.toggle_mask_mode()
                                elif chr_key == 'c':
                                    cfg.toggle_color_transfer_mode()
                                elif chr_key == 'v':
                                    cfg.toggle_super_resolution_mode()
                                elif chr_key == 'b':
                                    cfg.toggle_export_mask_alpha()
                                elif chr_key == 'n':
                                    cfg.toggle_sharpen_mode()

                            else:
                                if chr_key == 'y':
                                    cfg.add_blursharpen_amount(1 if not shift_pressed else 5)
                                elif chr_key == 'h':
                                    cfg.add_blursharpen_amount(-1 if not shift_pressed else -5)
                                elif chr_key == 's':
                                    cfg.toggle_add_source_image()
                                elif chr_key == 'v':
                                    cfg.toggle_super_resolution_mode()
                                elif chr_key == 'n':
                                    cfg.toggle_sharpen_mode()

                            if prev_cfg != cfg:
                                io.log_info ( cfg.to_string(cur_frame.frame_info.filename_short) )
                                cur_frame.is_done = False
                                cur_frame.is_shown = False
                    else:
                        if chr_key == ',' or chr_key == 'm':
                            self.process_remain_frames = False
                            go_prev_frame = True
                            go_prev_frame_overriding_cfg = chr_key == 'm'
                        elif chr_key == '.' or chr_key == '/':
                            self.process_remain_frames = False
                            go_next_frame = True
                            go_next_frame_overriding_cfg = chr_key == '/'
                        elif chr_key == '\r' or chr_key == '\n':
                            self.process_remain_frames = not self.process_remain_frames
                        elif chr_key == '-':
                            self.screen_manager.get_current().diff_scale(-0.1)
                        elif chr_key == '=':
                            self.screen_manager.get_current().diff_scale(0.1)


        if go_prev_frame:
            if cur_frame is None or cur_frame.is_done:
                if cur_frame is not None:
                    cur_frame.image = None

                if len(self.frames_done_idxs) > 0:
                    prev_frame = self.frames[self.frames_done_idxs.pop()]
                    self.frames_idxs.insert(0, prev_frame.idx)
                    prev_frame.is_shown = False
                    io.progress_bar_inc(-1)

                    if cur_frame is not None and go_prev_frame_overriding_cfg:
                        if prev_frame.cfg != cur_frame.cfg:
                            prev_frame.cfg = cur_frame.cfg.copy()
                            prev_frame.is_done = False

        elif go_next_frame:
            if cur_frame is not None and cur_frame.is_done:
                cur_frame.image = None
                cur_frame.is_shown = True
                self.frames_done_idxs.append(cur_frame.idx)
                self.frames_idxs.pop(0)
                io.progress_bar_inc(1)

                if len(self.frames_idxs) != 0:
                    next_frame = self.frames[ self.frames_idxs[0] ]

                    if go_next_frame_overriding_cfg:
                        f = self.frames
                        for i in range( next_frame.idx, len(self.frames) ):
                            f[i].cfg = None
                            f[i].is_shown = False

                    if next_frame.cfg is None or next_frame.is_shown == False: #next frame is never shown or override current cfg to next frames and the prefetches
                        for i in range( min(len(self.frames_idxs), self.prefetch_frame_count) ):
                            frame = self.frames[ self.frames_idxs[i] ]

                            if frame.cfg is None or frame.cfg != cur_frame.cfg:
                                frame.cfg = cur_frame.cfg.copy()
                                frame.is_done = False #initiate solve again


                    next_frame.is_shown = False

            if len(self.frames_idxs) == 0:
                self.process_remain_frames = False

        return (self.is_interactive and self.is_interactive_quitting) or \
               (not self.is_interactive and self.process_remain_frames == False)


    #override
    def on_data_return (self, host_dict, pf):
        frame = self.frames[pf.idx]
        frame.is_done = False
        frame.is_processing = False

    #override
    def on_result (self, host_dict, pf_sent, pf_result):
        frame = self.frames[pf_result.idx]
        frame.is_processing = False
        if frame.cfg == pf_result.cfg:
            frame.is_done = True
            frame.image = pf_result.image

    #override
    def get_data(self, host_dict):
        if self.is_interactive and self.is_interactive_quitting:
            return None

        for i in range ( min(len(self.frames_idxs), self.prefetch_frame_count) ):
            frame = self.frames[ self.frames_idxs[i] ]

            if not frame.is_done and not frame.is_processing and frame.cfg is not None:                
                frame.is_processing = True
                return ConvertSubprocessor.ProcessingFrame(idx=frame.idx,
                                                           cfg=frame.cfg.copy(),
                                                           prev_temporal_frame_infos=frame.prev_temporal_frame_infos,
                                                           frame_info=frame.frame_info,
                                                           next_temporal_frame_infos=frame.next_temporal_frame_infos,
                                                           output_filename=frame.output_filename,
                                                           need_return_image=True )

        return None

    #override
    def get_result(self):
        return 0

def main (args, device_args):
    io.log_info ("Running converter.\r\n")

    training_data_src_dir = args.get('training_data_src_dir', None)
    training_data_src_path = Path(training_data_src_dir) if training_data_src_dir is not None else None
    aligned_dir = args.get('aligned_dir', None)
    avaperator_aligned_dir = args.get('avaperator_aligned_dir', None)

    try:
        input_path = Path(args['input_dir'])
        output_path = Path(args['output_dir'])
        model_path = Path(args['model_dir'])

        if not input_path.exists():
            io.log_err('Input directory not found. Please ensure it exists.')
            return

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            io.log_err('Model directory not found. Please ensure it exists.')
            return

        is_interactive = io.input_bool ("Use interactive converter? (y/n skip:y) : ", True) if not io.is_colab() else False

        import models
        model = models.import_model( args['model_name'])(model_path, device_args=device_args, training_data_src_path=training_data_src_path)
        converter_session_filepath = model.get_strpath_storage_for_file('converter_session.dat')        
        predictor_func, predictor_input_shape, cfg = model.get_ConverterConfig()

        if not is_interactive:
            cfg.ask_settings()

        input_path_image_paths = Path_utils.get_image_paths(input_path)

        if cfg.type == ConverterConfig.TYPE_MASKED:
            if aligned_dir is None:
                io.log_err('Aligned directory not found. Please ensure it exists.')
                return

            aligned_path = Path(aligned_dir)
            if not aligned_path.exists():
                io.log_err('Aligned directory not found. Please ensure it exists.')
                return

            alignments = {}
            multiple_faces_detected = False
            aligned_path_image_paths = Path_utils.get_image_paths(aligned_path)
            for filepath in io.progress_bar_generator(aligned_path_image_paths, "Collecting alignments"):
                filepath = Path(filepath)

                if filepath.suffix == '.png':
                    dflimg = DFLPNG.load( str(filepath) )
                elif filepath.suffix == '.jpg':
                    dflimg = DFLJPG.load ( str(filepath) )
                else:
                    dflimg = None

                if dflimg is None:
                    io.log_err ("%s is not a dfl image file" % (filepath.name) )
                    continue

                source_filename_stem = Path( dflimg.get_source_filename() ).stem
                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments_ar = alignments[ source_filename_stem ]
                alignments_ar.append (dflimg.get_source_landmarks())
                if len(alignments_ar) > 1:
                    multiple_faces_detected = True

            if multiple_faces_detected:
                io.log_info ("Warning: multiple faces detected. Strongly recommended to process them separately.")

            frames = [ ConvertSubprocessor.Frame( frame_info=FrameInfo(filename=p, landmarks_list=alignments.get(Path(p).stem, None))) for p in input_path_image_paths ]

            if multiple_faces_detected:
                io.log_info ("Warning: multiple faces detected. Motion blur will not be used.")
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


        elif cfg.type == ConverterConfig.TYPE_FACE_AVATAR:
            filesdata = []
            for filepath in io.progress_bar_generator(input_path_image_paths, "Collecting info"):
                filepath = Path(filepath)

                if filepath.suffix == '.png':
                    dflimg = DFLPNG.load( str(filepath) )
                elif filepath.suffix == '.jpg':
                    dflimg = DFLJPG.load ( str(filepath) )
                else:
                    dflimg = None

                if dflimg is None:
                    io.log_err ("%s is not a dfl image file" % (filepath.name) )
                    continue
                filesdata += [ ( FrameInfo(filename=str(filepath), landmarks_list=[dflimg.get_landmarks()] ), dflimg.get_source_filename() ) ]

            filesdata = sorted(filesdata, key=operator.itemgetter(1)) #sort by filename
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

                frames.append ( ConvertSubprocessor.Frame(prev_temporal_frame_infos=prev_temporal_frame_infos,
                                                          frame_info=frame_info,
                                                          next_temporal_frame_infos=next_temporal_frame_infos) )

        if len(frames) == 0:
            io.log_info ("No frames to convert in input_dir.")
        else:
            ConvertSubprocessor (
                        is_interactive         = is_interactive,
                        converter_session_filepath = converter_session_filepath,
                        predictor_func         = predictor_func,
                        predictor_input_shape  = predictor_input_shape,                      
                        converter_config       = cfg,                        
                        frames                 = frames,
                        output_path            = output_path,
                        model_iter             = model.get_iter()
                    ).run()

        model.finalize()

    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()

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
