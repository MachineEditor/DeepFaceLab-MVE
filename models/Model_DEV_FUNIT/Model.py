from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib, FUNIT
from samplelib import *



class FUNITModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_face_type = 'f'
        if is_first_run:
            self.options['resolution'] = io.input_int("Resolution ( 128,224 ?:help skip:128) : ", 128, [128,224])
        else:
            self.options['resolution'] = self.options.get('resolution', 128)

        if is_first_run:
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="").lower()
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({4:16})

        resolution = self.options['resolution']
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF
        person_id_max_count = SampleGeneratorFace.get_person_id_max_count(self.training_data_src_path)

        
        self.model = FUNIT( face_type_str=FaceType.toString(face_type), 
                            batch_size=self.batch_size,
                            encoder_nf=64,
                            encoder_downs=2,
                            encoder_res_blk=2,
                            class_downs=4,
                            class_nf=64,
                            class_latent=64,
                            mlp_nf=256,
                            mlp_blks=2,
                            dis_nf=64,
                            dis_res_blks=10,
                            num_classes=person_id_max_count,
                            subpixel_decoder=True,
                            initialize_weights=self.is_first_run(),     
                            is_training=self.is_training_mode                       
                           )
                             
        if not self.is_first_run():
            self.load_weights_safe(self.model.get_model_filename_list())       
            
        if self.is_training_mode:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL if self.options['face_type'] == 'f' else t.FACE_TYPE_HALF
            
            output_sample_types=[ {'types': (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':128, 'normalize_tanh':True} ]

            self.set_training_data_generators ([
                        SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),

                        SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),

                        SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),

                        SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),
                    ])

    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list()

    #override
    def onSave(self):
        self.save_weights_safe(self.model.get_model_filename_list())

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        xa,la = generators_samples[0]
        xb,lb = generators_samples[1]
        
        G_loss, D_loss = self.model.train(xa,la,xb,lb)

        return ( ('G_loss', G_loss), ('D_loss', D_loss), )

    #override
    def onGetPreview(self, generators_samples):
        xa  = generators_samples[0][0]
        xb  = generators_samples[1][0]
        ta  = generators_samples[2][0]
        tb  = generators_samples[3][0]

        view_samples = min(4, xa.shape[0])

        lines_train = []
        lines_test = []

        for i in range(view_samples):

            s_xa = self.model.get_average_class_code([ xa[i:i+1] ])[0][None,...]
            s_xb = self.model.get_average_class_code([ xb[i:i+1] ])[0][None,...]

            s_ta = self.model.get_average_class_code([ ta[i:i+1] ])[0][None,...]
            s_tb = self.model.get_average_class_code([ tb[i:i+1] ])[0][None,...]

            xaxa = self.model.convert  ([ xa[i:i+1], s_xa  ] )[0][0]
            xbxb = self.model.convert  ([ xb[i:i+1], s_xb  ] )[0][0]
            xaxb = self.model.convert  ([ xa[i:i+1], s_xb  ] )[0][0]
            xbxa = self.model.convert  ([ xb[i:i+1], s_xa  ] )[0][0]

            tata = self.model.convert  ([ ta[i:i+1], s_ta  ] )[0][0]
            tbtb = self.model.convert  ([ tb[i:i+1], s_tb  ] )[0][0]
            tatb = self.model.convert  ([ ta[i:i+1], s_tb  ] )[0][0]
            tbta = self.model.convert  ([ tb[i:i+1], s_ta  ] )[0][0]

            line_train = [ xa[i], xaxa, xb[i], xbxb, xaxb, xbxa ]
            line_test =  [ ta[i], tata, tb[i], tbtb, tatb, tbta ]

            lines_train += [ np.concatenate([ np.clip(x/2+0.5,0,1) for x in line_train], axis=1) ]
            lines_test  += [ np.concatenate([ np.clip(x/2+0.5,0,1) for x in line_test ], axis=1) ]

        lines_train = np.concatenate ( lines_train, axis=0 )
        lines_test = np.concatenate ( lines_test, axis=0 )
        return [ ('TRAIN', lines_train ), ('TEST', lines_test) ]

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.model.convert ([ np.zeros ( (1, self.options['resolution'], self.options['resolution'], 3), dtype=np.float32 ), self.average_class_code ])
        else:
            bgr, = self.model.convert ([  face[np.newaxis,...]*2-1, self.average_class_code  ])
            return bgr[0] / 2 + 0.5

    #override
    def get_ConverterConfig(self):
        face_type = FaceType.FULL

        import converters
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), converters.ConverterConfigMasked(face_type=face_type,
                                     default_mode = 1,
                                     clip_hborder_mask_per=0.0625 if (face_type == FaceType.FULL) else 0,
                                    )


Model = FUNITModel
