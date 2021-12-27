import colorsys
import inspect
import multiprocessing
import operator
import os
import pickle
import shutil
import time
import datetime
from pathlib import Path
import yaml
from jsonschema import validate, ValidationError
import models

import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from samplelib import SampleGeneratorBase


class ModelBase(object):
    def __init__(self, is_training=False,
                       is_exporting=False,
                       saved_models_path=None,
                       training_data_src_path=None,
                       training_data_dst_path=None,
                       pretraining_data_path=None,
                       pretrained_model_path=None,
                       no_preview=False,
                       force_model_name=None,
                       force_gpu_idxs=None,
                       cpu_only=False,
                       debug=False,
                       force_model_class_name=None,
                       config_training_file=None,
                       auto_gen_config=False,
                       silent_start=False,
                       **kwargs):
        self.is_training = is_training
        self.is_exporting = is_exporting
        self.saved_models_path = saved_models_path
        self.training_data_src_path = training_data_src_path
        self.training_data_dst_path = training_data_dst_path
        self.pretraining_data_path = pretraining_data_path
        self.pretrained_model_path = pretrained_model_path
        self.config_training_file = config_training_file
        self.auto_gen_config = auto_gen_config
        self.config_file_path = None
        self.no_preview = no_preview
        self.debug = debug

        self.model_class_name = model_class_name = Path(inspect.getmodule(self).__file__).parent.name.rsplit("_", 1)[1]

        if force_model_class_name is None:
            if force_model_name is not None:
                self.model_name = force_model_name
            else:
                while True:
                    # gather all model dat files
                    saved_models_names = []
                    for filepath in pathex.get_file_paths(saved_models_path):
                        filepath_name = filepath.name
                        if filepath_name.endswith(f'{model_class_name}_data.dat'):
                            saved_models_names += [ (filepath_name.split('_')[0], os.path.getmtime(filepath)) ]

                    # sort by modified datetime
                    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True )
                    saved_models_names = [ x[0] for x in saved_models_names ]


                    if len(saved_models_names) != 0:
                        if silent_start:
                            self.model_name = saved_models_names[0]
                            io.log_info(f'Silent start: choosed model "{self.model_name}"')
                        else:
                            io.log_info ("Choose one of saved models, or enter a name to create a new model.")
                            io.log_info ("[r] : rename")
                            io.log_info ("[d] : delete")
                            io.log_info ("")
                            for i, model_name in enumerate(saved_models_names):
                                s = f"[{i}] : {model_name} "
                                if i == 0:
                                    s += "- latest"
                                io.log_info (s)

                            inp = io.input_str(f"", "0", show_default_value=False )
                            model_idx = -1
                            try:
                                model_idx = np.clip ( int(inp), 0, len(saved_models_names)-1 )
                            except:
                                pass

                            if model_idx == -1:
                                if len(inp) == 1:
                                    is_rename = inp[0] == 'r'
                                    is_delete = inp[0] == 'd'

                                    if is_rename or is_delete:
                                        if len(saved_models_names) != 0:

                                            if is_rename:
                                                name = io.input_str(f"Enter the name of the model you want to rename")
                                            elif is_delete:
                                                name = io.input_str(f"Enter the name of the model you want to delete")

                                            if name in saved_models_names:

                                                if is_rename:
                                                    new_model_name = io.input_str(f"Enter new name of the model")

                                                for filepath in pathex.get_paths(saved_models_path):
                                                    filepath_name = filepath.name

                                                    model_filename, remain_filename = filepath_name.split('_', 1)
                                                    if model_filename == name:

                                                        if is_rename:
                                                            new_filepath = filepath.parent / ( new_model_name + '_' + remain_filename )
                                                            filepath.rename (new_filepath)
                                                        elif is_delete:
                                                            filepath.unlink()
                                        continue

                                self.model_name = inp
                            else:
                                self.model_name = saved_models_names[model_idx]

                    else:
                        self.model_name = io.input_str(f"No saved models found. Enter a name of a new model", "new")
                        self.model_name = self.model_name.replace('_', ' ')
                    break


            self.model_name = self.model_name + '_' + self.model_class_name
        else:
            self.model_name = force_model_class_name

        self.iter = 0
        self.options = {}
        self.formatted_dictionary = {}
        self.options_show_override = {}
        self.loss_history = []
        self.sample_for_preview = None
        self.choosed_gpu_indexes = None

        model_data = {}
        # True if yaml conf file exists 
        self.config_file_exists = False
        # True if user chooses to read options external or internal conf file
        self.read_from_conf = False
        config_error = False
        #check if config_training_file mode is enabled
        if config_training_file is not None:
            if not Path(config_training_file).exists():
                 io.log_err(f"{config_training_file} does not exists, not using config!")
            else:
                self.config_file_path = Path(self.get_strpath_def_conf_file())
        elif self.auto_gen_config:
             self.config_file_path = Path(self.get_model_conf_path())

        if self.config_file_path is not None:
            self.read_from_conf = io.input_bool(
                f'Do you want to read training options from file?',
                True,
                'Read options from configuration file instead of asking one by one each option'
            ) if not silent_start else True

            # If user decides to read from external or internal conf file
            if self.read_from_conf:
                # Try to read dictionary from external of internal yaml file according
                # to the value of auto_gen_config
                self.options = self.read_from_config_file(self.config_file_path)
                # If options dict is empty, options will be loaded from dat file
                if self.options is None:
                    io.log_info(f"Config file validation error, check your config")
                    config_error = True
                elif not self.options.keys():
                    io.log_info(f"Configuration file doesn't exist. A standard configuration file will be created.")
                else:
                    io.log_info(f"Using config file from {self.config_file_path}")
                    self.config_file_exists = True

        self.model_data_path = Path( self.get_strpath_storage_for_file('data.dat') )
        if self.model_data_path.exists():
            io.log_info (f"Loading {self.model_name} model...")
            model_data = pickle.loads ( self.model_data_path.read_bytes() )
            self.iter = model_data.get('iter',0)
            if self.iter != 0:
                # read options from the .dat file only if the user chooses not to read options from the yaml file
                if not self.config_file_exists:
                    self.options = model_data['options']
                self.loss_history = model_data.get('loss_history', [])
                self.sample_for_preview = model_data.get('sample_for_preview', None)
                self.choosed_gpu_indexes = model_data.get('choosed_gpu_indexes', None)

        if self.is_first_run():
            io.log_info ("\nModel first run.")

        if silent_start:
            self.device_config = nn.DeviceConfig.BestGPU()
            io.log_info (f"Silent start: choosed device {'CPU' if self.device_config.cpu_only else self.device_config.devices[0].name}")
        else:
            self.device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or nn.ask_choose_device_idxs(suggest_best_multi_gpu=True)) \
                                if not cpu_only else nn.DeviceConfig.CPU()

        nn.initialize(self.device_config)

        ####
        self.default_options_path = saved_models_path / f'{self.model_class_name}_default_options.dat'
        self.default_options = {}
        if self.default_options_path.exists():
            try:
                self.default_options = pickle.loads ( self.default_options_path.read_bytes() )
            except:
                pass

        self.choose_preview_history = False
        self.batch_size = self.load_or_def_option('batch_size', 1)
        #####

        io.input_skip_pending()
        self.on_initialize_options()

        if self.is_first_run():
            # save as default options only for first run model initialize
            self.default_options_path.write_bytes( pickle.dumps (self.options) )

        # save config file
        if self.read_from_conf and not self.config_file_exists and not config_error and not self.config_file_path is None:
            self.save_config_file(self.config_file_path)

        self.session_name = self.options.get('session_name', "")
        self.autobackup_hour = self.options.get('autobackup_hour', 0)
        self.maximum_n_backups = self.options.get('maximum_n_backups', 24)
        self.write_preview_history = self.options.get('write_preview_history', False)
        self.target_iter = self.options.get('target_iter',0)
        self.random_flip = self.options.get('random_flip',True)
        self.random_src_flip = self.options.get('random_src_flip', False)
        self.random_dst_flip = self.options.get('random_dst_flip', True)
        self.retraining_samples = self.options.get('retraining_samples', False)
        
        self.on_initialize()
        self.options['batch_size'] = self.batch_size

        self.preview_history_writer = None
        if self.is_training:
            self.preview_history_path = self.saved_models_path / ( f'{self.get_model_name()}_history' )
            self.autobackups_path     = self.saved_models_path / ( f'{self.get_model_name()}_autobackups' )

            if self.write_preview_history or io.is_colab():
                if not self.preview_history_path.exists():
                    self.preview_history_path.mkdir(exist_ok=True)
                else:
                    if self.iter == 0:
                        for filename in pathex.get_image_paths(self.preview_history_path):
                            Path(filename).unlink()

            if self.generator_list is None:
                raise ValueError( 'You didnt set_training_data_generators()')
            else:
                for i, generator in enumerate(self.generator_list):
                    if not isinstance(generator, SampleGeneratorBase):
                        raise ValueError('training data generator is not subclass of SampleGeneratorBase')

            self.update_sample_for_preview(choose_preview_history=self.choose_preview_history)

            if self.autobackup_hour != 0:
                self.autobackup_start_time = time.time()

                if not self.autobackups_path.exists():
                    self.autobackups_path.mkdir(exist_ok=True)

        io.log_info( self.get_summary_text() )

    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        if self.sample_for_preview is None or choose_preview_history or force_new:
            if choose_preview_history and io.is_support_windows():
                wnd_name = "[p] - next. [space] - switch preview type. [enter] - confirm."
                io.log_info (f"Choose image for the preview history. {wnd_name}")
                io.named_window(wnd_name)
                io.capture_keys(wnd_name)
                choosed = False
                preview_id_counter = 0
                while not choosed:
                    self.sample_for_preview = self.generate_next_samples()
                    previews = self.get_history_previews()

                    io.show_image( wnd_name, ( previews[preview_id_counter % len(previews) ][1] *255).astype(np.uint8) )

                    while True:
                        key_events = io.get_key_events(wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)
                        if key == ord('\n') or key == ord('\r'):
                            choosed = True
                            break
                        elif key == ord(' '):
                            preview_id_counter += 1
                            break
                        elif key == ord('p'):
                            break

                        try:
                            io.process_messages(0.1)
                        except KeyboardInterrupt:
                            choosed = True

                io.destroy_window(wnd_name)
            else:
                self.sample_for_preview = self.generate_next_samples()

        try:
            self.get_history_previews()
        except:
            self.sample_for_preview = self.generate_next_samples()

        self.last_sample = self.sample_for_preview

    def load_or_def_option(self, name, def_value):
        options_val = self.options.get(name, None)
        if options_val is not None:
            return options_val

        def_opt_val = self.default_options.get(name, None)
        if def_opt_val is not None:
            return def_opt_val

        return def_value

    def ask_override(self):
        return self.is_training and self.iter != 0 and io.input_in_time ("Press enter in 2 seconds to override model settings.", 5 if io.is_colab() else 2 )

    def ask_session_name(self, default_value=""):
        default_session_name = self.options['session_name'] = self.load_or_def_option('session_name', default_value)
        self.options['session_name'] = io.input_str("Session name", default_session_name, help_message="String to refer back to in summary.txt and in autobackup foldername")

    def ask_autobackup_hour(self, default_value=0):
        default_autobackup_hour = self.options['autobackup_hour'] = self.load_or_def_option('autobackup_hour', default_value)
        self.options['autobackup_hour'] = io.input_int(f"Autobackup every N hour", default_autobackup_hour, add_info="0..24", help_message="Autobackup model files with preview every N hour. Latest backup is the last folder when sorted by name ascending located in model/<>_autobackups")

    def ask_maximum_n_backups(self, default_value=24):
        default_maximum_n_backups = self.options['maximum_n_backups'] = self.load_or_def_option('maximum_n_backups', default_value)
        self.options['maximum_n_backups'] = io.input_int(f"Maximum N backups", default_maximum_n_backups, help_message="Maximum amount of backups that are located in model/<>_autobackups. Inputting 0 here would allow it to autobackup as many times as it occurs.")

    def ask_write_preview_history(self, default_value=False):
        default_write_preview_history = self.load_or_def_option('write_preview_history', default_value)
        self.options['write_preview_history'] = io.input_bool(f"Write preview history", default_write_preview_history, help_message="Preview history will be written to <ModelName>_history folder.")

        if self.options['write_preview_history']:
            if io.is_support_windows():
                self.choose_preview_history = io.input_bool("Choose image for the preview history", False)
            elif io.is_colab():
                self.choose_preview_history = io.input_bool("Randomly choose new image for preview history", False, help_message="Preview image history will stay stuck with old faces if you reuse the same model on different celebs. Choose no unless you are changing src/dst to a new person")

    def ask_target_iter(self, default_value=0):
        default_target_iter = self.load_or_def_option('target_iter', default_value)
        self.options['target_iter'] = max(0, io.input_int("Target iteration", default_target_iter))

    def ask_random_flip(self):
        default_random_flip = self.load_or_def_option('random_flip', True)
        self.options['random_flip'] = io.input_bool("Flip faces randomly", default_random_flip, help_message="Predicted face will look more naturally without this option, but src faceset should cover all face directions as dst faceset.")
    
    def ask_random_src_flip(self):
        default_random_src_flip = self.load_or_def_option('random_src_flip', False)
        self.options['random_src_flip'] = io.input_bool("Flip SRC faces randomly", default_random_src_flip, help_message="Random horizontal flip SRC faceset. Covers more angles, but the face may look less naturally.")

    def ask_random_dst_flip(self):
        default_random_dst_flip = self.load_or_def_option('random_dst_flip', True)
        self.options['random_dst_flip'] = io.input_bool("Flip DST faces randomly", default_random_dst_flip, help_message="Random horizontal flip DST faceset. Makes generalization of src->dst better, if src random flip is not enabled.")

    def ask_batch_size(self, suggest_batch_size=None, range=None):
        default_batch_size = self.load_or_def_option('batch_size', suggest_batch_size or self.batch_size)

        batch_size = max(0, io.input_int("Batch_size", default_batch_size, valid_range=range, help_message="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually."))

        if range is not None:
            batch_size = np.clip(batch_size, range[0], range[1])

        self.options['batch_size'] = self.batch_size = batch_size

    def ask_retraining_samples(self, default_value=False):
        default_retraining_samples = self.load_or_def_option('retraining_samples', default_value)
        self.options['retraining_samples'] = io.input_bool("Retrain high loss samples", default_retraining_samples, help_message="Periodically retrains last 16 \"high-loss\" sample")


    #overridable
    def on_initialize_options(self):
        pass

    #overridable
    def on_initialize(self):
        '''
        initialize your models

        store and retrieve your model options in self.options['']

        check example
        '''
        pass

    #overridable
    def onSave(self):
        #save your models here
        pass

    #overridable
    def onTrainOneIter(self, sample, generator_list):
        #train your models here

        #return array of losses
        return ( ('loss_src', 0), ('loss_dst', 0) )

    #overridable
    def onGetPreview(self, sample, for_history=False, filenames=None):
        #you can return multiple previews
        #return [ ('preview_name',preview_rgb), ... ]
        return []

    #overridable if you want model name differs from folder name
    def get_model_name(self):
        return self.model_name

    #overridable , return [ [model, filename],... ]  list
    def get_model_filename_list(self):
        return []

    #overridable
    def get_MergerConfig(self):
        #return predictor_func, predictor_input_shape, MergerConfig() for the model
        raise NotImplementedError

    #overridable
    def get_config_schema_path(self):
        raise NotImplementedError

    #overridable
    def get_formatted_configuration_path(self):
        return "None"
        #raise NotImplementedError

    def get_pretraining_data_path(self):
        return self.pretraining_data_path

    def get_target_iter(self):
        return self.target_iter

    def is_reached_iter_goal(self):
        return self.target_iter != 0 and self.iter >= self.target_iter

    def get_previews(self):
        return self.onGetPreview ( self.last_sample, filenames=self.last_sample_filenames )
    
    def get_static_previews(self):
        return self.onGetPreview (self.sample_for_preview)

    def get_history_previews(self):
        return self.onGetPreview (self.sample_for_preview, for_history=True)

    def get_history_previews(self):
        return self.onGetPreview (self.sample_for_preview, for_history=True)

    def get_preview_history_writer(self):
        if self.preview_history_writer is None:
            self.preview_history_writer = PreviewHistoryWriter()
        return self.preview_history_writer

    def save(self):
        Path( self.get_summary_path() ).write_text( self.get_summary_text() )

        self.onSave()

        if self.auto_gen_config:
            path = Path(self.get_model_conf_path())
            self.save_config_file(path)

        model_data = {
            'iter': self.iter,
            'options': self.options,
            'loss_history': self.loss_history,
            'sample_for_preview' : self.sample_for_preview,
            'choosed_gpu_indexes' : self.choosed_gpu_indexes,
        }
        pathex.write_bytes_safe (self.model_data_path, pickle.dumps(model_data) )

        if self.autobackup_hour != 0:
            diff_hour = int ( (time.time() - self.autobackup_start_time) // 3600 )

            if diff_hour > 0 and diff_hour % self.autobackup_hour == 0:
                self.autobackup_start_time += self.autobackup_hour*3600
                self.create_backup()

    def __convert_type_write(self, value):
        if isinstance(value, (np.int32, np.float64, np.int64)):
            return value.item()
        else:
            return value

    def __update_nested_dict(self, nested_dict, key, val):
        if key in nested_dict:
            nested_dict[key] = self.__convert_type_write(val)
            return True
        for k, v in nested_dict.items():
              if isinstance(v, dict):
                  if self.__update_nested_dict(v, key, val):
                      return True
        return False

    def __iterate_read_dict(self, nested_dict, new_dict=None):
        if new_dict == None:
            new_dict = {}
        for k,v in nested_dict.items():
            if isinstance(v, dict):
                new_dict.update(self.__iterate_read_dict(v, new_dict))
            else:            
                new_dict[k] = v
        return new_dict

    def read_from_config_file(self, filepath, keep_nested=False, validation=True):
        """
        Reads options from configuration yaml file.

        Args:
            filepath (str|Path): Path where to read configuration file.
            keep_nested (bool, optional): If false dictionary is keep nested, otherwise not. Defaults to False.
            validation (bool, optional): If true dictionary is valideted. Defaults to True.

        Returns:
            [dict]: A dictionary of options.
        """
        #fun = self.get_strpath_configuration_path if not auto_gen else self.get_model_conf_path
        data = {}
        try:
            with open(filepath, 'r') as file, open(self.get_config_schema_path(), 'r') as schema:
                data = yaml.safe_load(file)
                if not keep_nested:
                    data = self.__iterate_read_dict(data)
                if validation:
                    validate(data, yaml.safe_load(schema))
        except FileNotFoundError:
            return {}
        except ValidationError as ve:
            io.log_err(f"{ve}")
            return None

        return data

    def save_config_file(self, filepath):
        """
        Saves options to configuration yaml file

        Args:
            filepath (str|Path): Path where to save configuration file.
        """

        formatted_dict = self.read_from_config_file(self.get_formatted_configuration_path(), keep_nested=True, validation=False)

        for key, value in self.options.items():
            if not self.__update_nested_dict(formatted_dict, key, value):
                formatted_dict[key] = self.__convert_type_write(value)

        try:
            with open(filepath, 'w') as file:
                yaml.dump(formatted_dict, file, sort_keys=False)
        except OSError as exception:
            io.log_info('Impossible to write YAML configuration file -> ', exception)

    def create_backup(self):
        io.log_info ("Creating backup...", end='\r')

        if not self.autobackups_path.exists():
            self.autobackups_path.mkdir(exist_ok=True)

        bckp_filename_list = [ self.get_strpath_storage_for_file(filename) for _, filename in self.get_model_filename_list() ]
        bckp_filename_list += [ str(self.get_summary_path()), str(self.model_data_path) ]

        # Create new backup
        session_suffix = f'_{self.session_name}' if self.session_name else ''
        idx_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + session_suffix
        idx_backup_path = self.autobackups_path / idx_str
        idx_backup_path.mkdir()
        for filename in bckp_filename_list:
            shutil.copy(str(filename), str(idx_backup_path / Path(filename).name))\

        previews = self.get_previews()

        # Generate previews and save in new backup
        plist = []
        for i in range(len(previews)):
            name, bgr = previews[i]
            plist += [ (bgr, idx_backup_path / ( ('preview_%s.jpg') % (name))  )  ]

        if len(plist) != 0:
            self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

        # Check if we've exceeded the max number of backups
        if self.maximum_n_backups != 0:
            all_backups = sorted([x for x in self.autobackups_path.iterdir() if x.is_dir()])
            while len(all_backups) > self.maximum_n_backups:
                oldest_backup = all_backups.pop(0)
                pathex.delete_all_files(oldest_backup)
                oldest_backup.rmdir()

    def debug_one_iter(self):
        images = []
        for generator in self.generator_list:
            for i,batch in enumerate(next(generator)):
                if len(batch.shape) == 4:
                    images.append( batch[0] )

        return imagelib.equalize_and_stack_square (images)

    def generate_next_samples(self):
        sample = []
        sample_filenames = []
        for generator in self.generator_list:
            if generator.is_initialized():
                batch = generator.generate_next()
                if type(batch) is tuple:
                    sample.append ( batch[0] )
                    sample_filenames.append( batch[1] )
                else:
                    sample.append ( batch )
            else:
                sample.append ( [] )
        self.last_sample = sample
        self.last_sample_filenames = sample_filenames
        return sample

    #overridable
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % 10 == 0) or (io.is_colab() and self.iter % 100 == 0)

    def train_one_iter(self):

        iter_time = time.time()
        losses = self.onTrainOneIter()
        iter_time = time.time() - iter_time

        self.loss_history.append ( [float(loss[1]) for loss in losses] )

        if self.should_save_preview_history():
            plist = []

            if io.is_colab():
                previews = self.get_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [ (bgr, self.get_strpath_storage_for_file('preview_%s.jpg' % (name) ) ) ]

            if self.write_preview_history:
                previews = self.get_history_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    path = self.preview_history_path / name
                    plist += [ ( bgr, str ( path / ( f'{self.iter:07d}.jpg') ) ) ]
                    if not io.is_colab():
                        plist += [ ( bgr, str ( path / ( '_last.jpg' ) )) ]

            if len(plist) != 0:
                self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

        self.iter += 1

        return self.iter, iter_time

    def pass_one_iter(self):
        self.generate_next_samples()

    def finalize(self):
        nn.close_session()

    def is_first_run(self):
        return self.iter == 0

    def is_debug(self):
        return self.debug

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_iter(self):
        return self.iter

    def set_iter(self, iter):
        self.iter = iter
        self.loss_history = self.loss_history[:iter]

    def get_loss_history(self):
        return self.loss_history

    def set_training_data_generators (self, generator_list):
        self.generator_list = generator_list

    def get_training_data_generators (self):
        return self.generator_list

    def get_model_root_path(self):
        return self.saved_models_path

    def get_strpath_storage_for_file(self, filename):
        return str( self.saved_models_path / ( self.get_model_name() + '_' + filename) )

    def get_strpath_configuration_path(self):
        return str(self.config_file_path)

    def get_strpath_def_conf_file(self):
        if Path(self.config_training_file).is_file():
            return str(Path(self.config_training_file))
        elif Path(self.config_training_file).is_dir(): #left for workspace bat backwards compatibility 
            return str(Path(self.config_training_file) / 'def_conf_file.yaml')
        else: return None
            
    def get_summary_path(self):
        return self.get_strpath_storage_for_file('summary.txt')

    def get_model_conf_path(self):
        return self.get_strpath_storage_for_file('configuration_file.yaml')

    def get_summary_text(self):
        visible_options = self.options.copy()
        visible_options.update(self.options_show_override)

        ###Generate text summary of model hyperparameters
        #Find the longest key name and value string. Used as column widths.
        width_name = max([len(k) for k in visible_options.keys()] + [17]) + 1 # Single space buffer to left edge. Minimum of 17, the length of the longest static string used "Current iteration"
        width_value = max([len(str(x)) for x in visible_options.values()] + [len(str(self.get_iter())), len(self.get_model_name())]) + 1 # Single space buffer to right edge
        if len(self.device_config.devices) != 0: #Check length of GPU names
            width_value = max([len(device.name)+1 for device in self.device_config.devices] + [width_value])
        width_total = width_name + width_value + 2 #Plus 2 for ": "

        summary_text = []
        summary_text += [f'=={" Model Summary ":=^{width_total}}=='] # Model/status summary
        summary_text += [f'=={" "*width_total}==']
        summary_text += [f'=={"Model name": >{width_name}}: {self.get_model_name(): <{width_value}}=='] # Name
        summary_text += [f'=={" "*width_total}==']
        summary_text += [f'=={"Current iteration": >{width_name}}: {str(self.get_iter()): <{width_value}}=='] # Iter
        summary_text += [f'=={" "*width_total}==']

        summary_text += [f'=={" Model Options ":-^{width_total}}=='] # Model options
        summary_text += [f'=={" "*width_total}==']
        for key in visible_options.keys():
            summary_text += [f'=={key: >{width_name}}: {str(visible_options[key]): <{width_value}}=='] # visible_options key/value pairs
        summary_text += [f'=={" "*width_total}==']

        summary_text += [f'=={" Running On ":-^{width_total}}=='] # Training hardware info
        summary_text += [f'=={" "*width_total}==']
        if len(self.device_config.devices) == 0:
            summary_text += [f'=={"Using device": >{width_name}}: {"CPU": <{width_value}}=='] # cpu_only
        else:
            for device in self.device_config.devices:
                summary_text += [f'=={"Device index": >{width_name}}: {device.index: <{width_value}}=='] # GPU hardware device index
                summary_text += [f'=={"Name": >{width_name}}: {device.name: <{width_value}}=='] # GPU name
                vram_str = f'{device.total_mem_gb:.2f}GB' # GPU VRAM - Formated as #.## (or ##.##)
                summary_text += [f'=={"VRAM": >{width_name}}: {vram_str: <{width_value}}==']
        summary_text += [f'=={" "*width_total}==']
        summary_text += [f'=={"="*width_total}==']
        summary_text = "\n".join (summary_text)
        return summary_text

    @staticmethod
    def get_loss_history_preview(loss_history, iter, w, c, lh_height=100):
        loss_history = np.array (loss_history.copy())

        lh_img = np.ones ( (lh_height,w,c) ) * 0.1

        if len(loss_history) != 0:
            loss_count = len(loss_history[0])
            lh_len = len(loss_history)

            l_per_col = lh_len / w
            plist_max = [   [   max (0.0, loss_history[int(col*l_per_col)][p],
                                                *[  loss_history[i_ab][p]
                                                    for i_ab in range( int(col*l_per_col), int((col+1)*l_per_col) )
                                                ]
                                    )
                                for p in range(loss_count)
                            ]
                            for col in range(w)
                        ]

            plist_min = [   [   min (plist_max[col][p], loss_history[int(col*l_per_col)][p],
                                                *[  loss_history[i_ab][p]
                                                    for i_ab in range( int(col*l_per_col), int((col+1)*l_per_col) )
                                                ]
                                    )
                                for p in range(loss_count)
                            ]
                            for col in range(w)
                        ]

            plist_abs_max = np.mean(loss_history[ len(loss_history) // 5 : ]) * 2

            for col in range(0, w):
                for p in range(0,loss_count):
                    point_color = [1.0]*c
                    point_color[0:3] = colorsys.hsv_to_rgb ( p * (1.0/loss_count), 1.0, 1.0 )

                    ph_max = int ( (plist_max[col][p] / plist_abs_max) * (lh_height-1) )
                    ph_max = np.clip( ph_max, 0, lh_height-1 )

                    ph_min = int ( (plist_min[col][p] / plist_abs_max) * (lh_height-1) )
                    ph_min = np.clip( ph_min, 0, lh_height-1 )

                    for ph in range(ph_min, ph_max+1):
                        lh_img[ (lh_height-ph-1), col ] = point_color

        lh_lines = 5
        lh_line_height = (lh_height-1)/lh_lines
        for i in range(0,lh_lines+1):
            lh_img[ int(i*lh_line_height), : ] = (0.8,)*c

        last_line_t = int((lh_lines-1)*lh_line_height)
        last_line_b = int(lh_lines*lh_line_height)

        lh_text = 'Iter: %d' % (iter) if iter != 0 else ''

        lh_img[last_line_t:last_line_b, 0:w] += imagelib.get_text_image (  (last_line_b-last_line_t,w,c), lh_text, color=[0.8]*c )
        return lh_img

class PreviewHistoryWriter():
    def __init__(self):
        self.sq = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.process, args=( self.sq, ))
        self.p.daemon = True
        self.p.start()

    def process(self, sq):
        while True:
            while not sq.empty():
                plist, loss_history, iter = sq.get()

                preview_lh_cache = {}
                for preview, filepath in plist:
                    filepath = Path(filepath)
                    i = (preview.shape[1], preview.shape[2])

                    preview_lh = preview_lh_cache.get(i, None)
                    if preview_lh is None:
                        preview_lh = ModelBase.get_loss_history_preview(loss_history, iter, preview.shape[1], preview.shape[2])
                        preview_lh_cache[i] = preview_lh

                    img = (np.concatenate ( [preview_lh, preview], axis=0 ) * 255).astype(np.uint8)

                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2_imwrite (filepath, img )

            time.sleep(0.01)

    def post(self, plist, loss_history, iter):
        self.sq.put ( (plist, loss_history, iter) )

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)
