import multiprocessing
import operator
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType, XSegNet
from models import ModelBase
from samplelib import *

class XSegModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_model_class_name='XSeg', **kwargs)

    #override
    def on_initialize_options(self):
        ask_override = self.ask_override()

        if not self.is_first_run() and ask_override:
            if io.input_bool(f"Restart training?", False, help_message="Reset model weights and start training from scratch."):
                self.set_iter(0)

        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'wf')
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        if self.is_first_run():
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Choose the same as your deepfake model.").lower()

        if self.is_first_run() or ask_override:
            self.ask_batch_size(4, range=[2,16])
            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain)
        
        if not self.is_exporting and (self.options['pretrain'] and self.get_pretraining_data_path() is None):
            raise Exception("pretraining_data_path is not defined")
            
        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)
        
    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        self.model_data_format = "NCHW" if self.is_exporting or (len(device_config.devices) != 0 and not self.is_debug()) else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        self.resolution = resolution = 256


        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]
        
            
        place_model_on_cpu = len(devices) == 0
        models_opt_device = '/CPU:0' if place_model_on_cpu else nn.tf_default_device_name

        bgr_shape = nn.get4Dshape(resolution,resolution,3)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        # Initializing model classes
        self.model = XSegNet(name='XSeg',
                               resolution=resolution,
                               load_weights=not self.is_first_run(),
                               weights_file_root=self.get_model_root_path(),
                               training=True,
                               place_model_on_cpu=place_model_on_cpu,
                               optimizer=nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt'),
                               data_format=nn.data_format)
        
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)
            
        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            # Compute losses per GPU
            gpu_pred_list = []

            gpu_losses = []
            gpu_loss_gvs = []

            for gpu_id in range(gpu_count):
                with tf.device(f'/{devices[gpu_id].tf_dev_type}:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_input_t       = self.model.input_t [batch_slice,:,:,:]
                        gpu_target_t      = self.model.target_t [batch_slice,:,:,:]

                    # process model tensors
                    gpu_pred_logits_t, gpu_pred_t = self.model.flow(gpu_input_t, pretrain=self.pretrain)
                    gpu_pred_list.append(gpu_pred_t)
                    
                    
                    if self.pretrain:
                        # Structural loss
                        gpu_loss =  tf.reduce_mean (5*nn.dssim(gpu_target_t, gpu_pred_t, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                        gpu_loss += tf.reduce_mean (5*nn.dssim(gpu_target_t, gpu_pred_t, max_val=1.0, filter_size=int(resolution/23.2)), axis=[1])
                        # Pixel loss
                        gpu_loss += tf.reduce_mean (10*tf.square(gpu_target_t-gpu_pred_t), axis=[1,2,3])
                    else:
                        gpu_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=gpu_target_t, logits=gpu_pred_logits_t), axis=[1,2,3])
                      
                    gpu_losses += [gpu_loss]

                    gpu_loss_gvs += [ nn.gradients ( gpu_loss, self.model.get_weights() ) ]


            # Average losses and gradients, and create optimizer update ops
            #with tf.device(f'/CPU:0'): # Temporary fix. Unknown bug with training freeze starts from 2.4.0, but 2.3.1 was ok
            with tf.device (models_opt_device):
                pred = tf.concat(gpu_pred_list, 0)
                loss = tf.concat(gpu_losses, 0)
                loss_gv_op = self.model.opt.get_update_op (nn.average_gv_list (gpu_loss_gvs))


            # Initializing training and view functions
            if self.pretrain:
                def train(input_np, target_np):
                    l, _ = nn.tf_sess.run ( [loss, loss_gv_op], feed_dict={self.model.input_t :input_np, self.model.target_t :target_np})
                    return l
            else:
                def train(input_np, target_np):
                    l, _ = nn.tf_sess.run ( [loss, loss_gv_op], feed_dict={self.model.input_t :input_np, self.model.target_t :target_np })
                    return l
            self.train = train

            def view(input_np):
                return nn.tf_sess.run ( [pred], feed_dict={self.model.input_t :input_np})
            self.view = view

            # initializing sample generators
            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_dst_generators_count = cpu_count // 2
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            
            if self.pretrain:
                pretrain_gen = SampleGeneratorFace(self.get_pretraining_data_path(), debug=self.is_debug(), batch_size=self.get_batch_size(),
                                    sample_process_options=SampleProcessor.Options(random_flip=True),
                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                            {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},                                                            
                                                          ],
                                    uniform_yaw_distribution=False,
                                    generators_count=cpu_count )
                self.set_training_data_generators ([pretrain_gen])
            else:   
                srcdst_generator = SampleGeneratorFaceXSeg([self.training_data_src_path, self.training_data_dst_path],
                                                            debug=self.is_debug(),
                                                            batch_size=self.get_batch_size(),
                                                            resolution=resolution,
                                                            face_type=self.face_type,
                                                            generators_count=src_dst_generators_count,
                                                            data_format=nn.data_format)

                src_generator = SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                                    sample_process_options=SampleProcessor.Options(random_flip=False),
                                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,  'warp':False, 'transform':False, 'channel_type' : SampleProcessor.ChannelType.BGR, 'border_replicate':False, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                                        ],
                                                    generators_count=src_generators_count,
                                                    raise_on_no_data=False )
                dst_generator = SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                                    sample_process_options=SampleProcessor.Options(random_flip=False),
                                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,  'warp':False, 'transform':False, 'channel_type' : SampleProcessor.ChannelType.BGR, 'border_replicate':False, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                                        ],
                                                    generators_count=dst_generators_count,
                                                    raise_on_no_data=False )

                self.set_training_data_generators ([srcdst_generator, src_generator, dst_generator])

    #override
    def get_model_filename_list(self):
        return self.model.model_filename_list

    #override
    def onSave(self):
        self.model.save_weights()

    #override
    def onTrainOneIter(self):
        image_np, target_np = self.generate_next_samples()[0]
        loss = self.train (image_np, target_np)
        
        return ( ('loss', np.mean(loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )
        
        if self.pretrain:
            srcdst_samples, = samples       
            image_np, mask_np = srcdst_samples     
        else:
            srcdst_samples, src_samples, dst_samples = samples
            image_np, mask_np = srcdst_samples

        I, M, IM, = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([image_np,mask_np] + self.view (image_np) ) ]
        M, IM, = [ np.repeat (x, (3,), -1) for x in [M, IM] ]

        green_bg = np.tile( np.array([0,1,0], dtype=np.float32)[None,None,...], (self.resolution,self.resolution,1) )

        result = []
        st = []
        for i in range(n_samples):
            if self.pretrain:
                ar = I[i], IM[i]
            else:
                ar = I[i]*M[i]+0.5*I[i]*(1-M[i])+0.5*green_bg*(1-M[i]), IM[i], I[i]*IM[i]+0.5*I[i]*(1-IM[i]) + 0.5*green_bg*(1-IM[i])
            st.append ( np.concatenate ( ar, axis=1) )
        result += [ ('XSeg training faces', np.concatenate (st, axis=0 )), ]

        if not self.pretrain and len(src_samples) != 0:
            src_np, = src_samples


            D, DM, = [ np.clip(nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([src_np] + self.view (src_np) ) ]
            DM, = [ np.repeat (x, (3,), -1) for x in [DM] ]

            st = []
            for i in range(n_samples):
                ar = D[i], DM[i], D[i]*DM[i] + 0.5*D[i]*(1-DM[i]) + 0.5*green_bg*(1-DM[i])
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('XSeg src faces', np.concatenate (st, axis=0 )), ]

        if not self.pretrain and len(dst_samples) != 0:
            dst_np, = dst_samples


            D, DM, = [ np.clip(nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([dst_np] + self.view (dst_np) ) ]
            DM, = [ np.repeat (x, (3,), -1) for x in [DM] ]

            st = []
            for i in range(n_samples):
                ar = D[i], DM[i], D[i]*DM[i]  + 0.5*D[i]*(1-DM[i]) + 0.5*green_bg*(1-DM[i])
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('XSeg dst faces', np.concatenate (st, axis=0 )), ]

        return result
        
    def export_dfm (self):
        output_path = self.get_strpath_storage_for_file(f'model.onnx')
        io.log_info(f'Dumping .onnx to {output_path}')
        tf = nn.tf
        
        with tf.device (nn.tf_default_device_name):
            input_t = tf.placeholder (nn.floatx, (None, self.resolution, self.resolution, 3), name='in_face')
            input_t = tf.transpose(input_t, (0,3,1,2))
            _, pred_t = self.model.flow(input_t)
            pred_t = tf.transpose(pred_t, (0,2,3,1))
            
        tf.identity(pred_t, name='out_mask')
        
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            nn.tf_sess, 
            tf.get_default_graph().as_graph_def(), 
            ['out_mask']
        ) 
        
        import tf2onnx
        with tf.device("/CPU:0"):
            model_proto, _ = tf2onnx.convert._convert_common(
                output_graph_def,
                name='XSeg',
                input_names=['in_face:0'],
                output_names=['out_mask:0'],
                opset=13,
                output_path=output_path)
                
Model = XSegModel