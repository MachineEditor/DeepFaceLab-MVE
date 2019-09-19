import numpy as np

from facelib import FaceType
from interact import interact as io
from models import ModelBase
from nnlib import nnlib, FUNIT
from samplelib import *

class TrueFaceModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_resolution = 128
        default_face_type = 'f'

        if is_first_run:
            resolution = self.options['resolution'] = io.input_int(f"Resolution ( 64-256 ?:help skip:{default_resolution}) : ", default_resolution, help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
            resolution = np.clip (resolution, 64, 256)
            while np.modf(resolution / 16)[0] != 0.0:
                resolution -= 1
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)

        if is_first_run:
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="").lower()
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)

    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({2:1,3:1,4:4,5:8,6:16})

        resolution = self.options['resolution']
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

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
                            num_classes=2,
                            subpixel_decoder=True,
                            initialize_weights=self.is_first_run(),
                            is_training=self.is_training_mode
                           )

        if not self.is_first_run():
            self.load_weights_safe(self.model.get_model_filename_list())

        t = SampleProcessor.Types
        face_type = t.FACE_TYPE_FULL if self.options['face_type'] == 'f' else t.FACE_TYPE_HALF
        if self.is_training_mode:

            output_sample_types=[ {'types': (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':resolution, 'normalize_tanh':True},
                                 ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True),
                        output_sample_types=output_sample_types )
                   ])
        else:
            generator = SampleGeneratorFace(self.training_data_src_path, batch_size=1,
                                sample_process_options=SampleProcessor.Options(),
                                output_sample_types=[ {'types': (t.IMG_SOURCE, face_type, t.MODE_BGR), 'resolution':resolution, 'normalize_tanh':True} ] )

            io.log_info("Calculating average src face style...")
            codes = []
            for i in io.progress_bar_generator(range(generator.get_total_sample_count())):
                codes += self.model.get_average_class_code( generator.generate_next() )

            self.average_class_code = np.mean ( np.array(codes), axis=0 )[None,...]


    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list()

    #override
    def onSave(self):
        self.save_weights_safe(self.model.get_model_filename_list())

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        bs = self.batch_size
        lbs = bs // 2
        hbs = bs - lbs

        src, = generators_samples[0]
        dst, = generators_samples[1]

        xa  = np.concatenate ( [src[0:lbs], dst[0:lbs]], axis=0 )

        la = np.concatenate ( [ np.array ([0]*lbs, np.int32),
                                np.array ([1]*lbs, np.int32) ] )

        xb = np.concatenate ( [src[lbs:], dst[lbs:]], axis=0 )

        lb = np.concatenate ( [ np.array ([0]*hbs, np.int32),
                                np.array ([1]*hbs, np.int32) ] )

        rnd_list = np.arange(lbs*2)
        np.random.shuffle(rnd_list)
        xa  = xa[rnd_list,...]
        la = la[rnd_list,...]
        la = la[...,None]

        rnd_list = np.arange(hbs*2)
        np.random.shuffle(rnd_list)
        xb = xb[rnd_list,...]
        lb = lb[rnd_list,...]
        lb = lb[...,None]

        G_loss, D_loss = self.model.train(xa,la,xb,lb)

        return ( ('G_loss', G_loss), ('D_loss', D_loss), )

    #override
    def onGetPreview(self, generators_samples):
        xa  = generators_samples[0][0]
        xb  = generators_samples[1][0]

        view_samples = min(4, xa.shape[0])


        s_xa_mean = self.model.get_average_class_code([xa])[0][None,...]
        s_xb_mean = self.model.get_average_class_code([xb])[0][None,...]

        s_xab_mean = self.model.get_average_class_code([ np.concatenate( [xa,xb], axis=0) ])[0][None,...]

        lines = []

        for i in range(view_samples):
            xaxa, = self.model.convert  ([ xa[i:i+1], s_xa_mean  ] )
            xbxb, = self.model.convert  ([ xb[i:i+1], s_xb_mean  ] )
            xbxa, = self.model.convert  ([ xb[i:i+1], s_xa_mean  ] )

            xa_i,xb_i,xaxa,xbxb,xbxa = [ np.clip(x/2+0.5, 0, 1) for x in [xa[i], xb[i], xaxa[0],xbxb[0],xbxa[0]] ]

            lines += [ np.concatenate( (xa_i, xaxa, xb_i, xbxb, xbxa), axis=1) ]

        r = np.concatenate ( lines, axis=0 )
        return [ ('TrueFace', r ) ]

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

Model = TrueFaceModel
