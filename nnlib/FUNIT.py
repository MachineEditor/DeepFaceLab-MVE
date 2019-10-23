from pathlib import Path

import numpy as np

from interact import interact as io
from nnlib import nnlib

"""
My port of FUNIT: Few-Shot Unsupervised Image-to-Image Translation to pure keras.
original repo: https://github.com/NVlabs/FUNIT/
"""
class FUNIT(object):
    VERSION = 1
    def __init__ (self, face_type_str,
                        batch_size,
                        encoder_nf=64,
                        encoder_downs=2,
                        encoder_res_blk=2,
                        class_downs=4,
                        class_nf=64,
                        class_latent=64,
                        mlp_blks=2,
                        dis_nf=64,
                        dis_res_blks=10,
                        num_classes=2,
                        subpixel_decoder=True,
                        initialize_weights=True,

                        load_weights_locally=False,
                        weights_file_root=None,

                        is_training=True,
                        tf_cpu_mode=0,
                        ):
        exec( nnlib.import_all(), locals(), globals() )

        self.batch_size = batch_size
        bgr_shape = (None, None, 3)
        label_shape = (1,)

        self.enc_content = modelify ( FUNIT.ContentEncoderFlow(downs=encoder_downs, nf=encoder_nf, n_res_blks=encoder_res_blk) ) ( Input(bgr_shape) )
        self.enc_class_model = modelify ( FUNIT.ClassModelEncoderFlow(downs=class_downs, nf=class_nf, latent_dim=class_latent) ) ( Input(bgr_shape) )
        self.decoder     = modelify ( FUNIT.DecoderFlow(ups=encoder_downs, n_res_blks=encoder_res_blk, mlp_blks=mlp_blks, subpixel_decoder=subpixel_decoder  ) ) \
                             ( [ Input(K.int_shape(self.enc_content.outputs[0])[1:], name="decoder_input_1"),
                                 Input(K.int_shape(self.enc_class_model.outputs[0])[1:], name="decoder_input_2")
                               ] )

        self.dis = modelify ( FUNIT.DiscriminatorFlow(nf=dis_nf, n_res_blks=dis_res_blks, num_classes=num_classes) ) (Input(bgr_shape))

        self.G_opt = RMSprop(lr=0.0001, decay=0.0001, tf_cpu_mode=tf_cpu_mode)
        self.D_opt = RMSprop(lr=0.0001, decay=0.0001, tf_cpu_mode=tf_cpu_mode)

        xa = Input(bgr_shape, name="xa")
        la = Input(label_shape, dtype="int32", name="la")

        xb = Input(bgr_shape, name="xb")
        lb = Input(label_shape, dtype="int32", name="lb")

        s_xa_one = Input( (  K.int_shape(self.enc_class_model.outputs[0])[-1],), name="s_xa_input")

        c_xa = self.enc_content(xa)

        s_xa = self.enc_class_model(xa)
        s_xb = self.enc_class_model(xb)

        s_xa_mean = K.mean(s_xa, axis=0)

        xr = self.decoder ([c_xa,s_xa])
        xt = self.decoder ([c_xa,s_xb])
        xr_one = self.decoder ([c_xa,s_xa_one])

        d_xr, d_xr_feat = self.dis(xr)
        d_xt, d_xt_feat = self.dis(xt)

        d_xa, d_xa_feat = self.dis(xa)
        d_xb, d_xb_feat = self.dis(xb)

        def dis_gather(x,l):
            tensors = []
            for i in range(self.batch_size):
                t = x[i:i+1,:,:, l[i,0]]
                tensors += [t]
            return tensors

        def dis_gather_batch_mean(x,l, func=None):
            x_shape = K.shape(x)
            b,h,w,c = x_shape[0],x_shape[1],x_shape[2],x_shape[3]
            b,h,w,c = [ K.cast(x, K.floatx()) for x in [b,h,w,c] ]

            tensors = dis_gather(x,l)
            if func is not None:
                tensors = [func(t) for t in tensors]

            return K.sum(tensors, axis=[1,2,3]) / (h*w)

        def dis_gather_mean(x,l, func=None, acc_func=None):
            x_shape = K.shape(x)
            b,h,w,c = x_shape[0],x_shape[1],x_shape[2],x_shape[3]
            b,h,w,c = [ K.cast(x, K.floatx()) for x in [b,h,w,c] ]

            tensors = dis_gather(x,l)

            if acc_func is not None:
                acc = []
                for t in tensors:
                    acc += [ K.sum( K.cast( acc_func(t), K.floatx() )) ]
                acc = K.cast( K.sum(acc), K.floatx() ) / (b*h*w)
            else:
                acc = None

            if func is not None:
                tensors = [func(t) for t in tensors]
            
            return K.sum(tensors, axis=[1,2,3] ) / (h*w), acc

        d_xr_la, d_xr_la_acc = dis_gather_mean(d_xr, la, acc_func=lambda x: x >= 0)
        d_xt_lb, d_xt_lb_acc = dis_gather_mean(d_xt, lb, acc_func=lambda x: x >= 0)

        d_xb_lb = dis_gather_batch_mean(d_xb, lb)

        d_xb_lb_real, d_xb_lb_real_acc = dis_gather_mean(d_xb, lb, lambda x: K.relu(1.0-x), acc_func=lambda x: x >= 0)
        d_xt_lb_fake, d_xt_lb_fake_acc = dis_gather_mean(d_xt, lb, lambda x: K.relu(1.0+x), acc_func=lambda x: x < 0)
        

        G_c_rec = K.mean(K.abs(K.mean(d_xr_feat, axis=[1,2]) - K.mean(d_xa_feat, axis=[1,2])), axis=1 ) #* 1.0
        G_m_rec = K.mean(K.abs(K.mean(d_xt_feat, axis=[1,2]) - K.mean(d_xb_feat, axis=[1,2])), axis=1 ) #* 1.0
        G_x_rec = 0.1 * K.mean(K.abs(xr-xa), axis=[1,2,3])

        G_loss = (-d_xr_la-d_xt_lb)*0.5 + G_x_rec + G_c_rec + G_m_rec

        G_weights = self.enc_class_model.trainable_weights + self.enc_content.trainable_weights + self.decoder.trainable_weights
        ######

        D_real = d_xb_lb_real #1.0 *
        D_fake = d_xt_lb_fake #1.0 *
        
        l_reg = 10 * K.sum( K.gradients( d_xb_lb, xb )[0] ** 2 , axis=[1,2,3] ) #/ self.batch_size )

        D_loss = D_real + D_fake + l_reg

        D_weights = self.dis.trainable_weights

        self.G_train = K.function ([xa, la, xb, lb],[K.mean(G_loss)], self.G_opt.get_updates(G_loss, G_weights) )

        self.D_train = K.function ([xa, la, xb, lb],[K.mean(D_loss)], self.D_opt.get_updates(D_loss, D_weights) )
        self.get_average_class_code = K.function ([xa],[s_xa_mean])

        self.G_convert = K.function  ([xa,s_xa_one],[xr_one])

        if initialize_weights:
            #gather weights from layers for initialization
            weights_list = []
            for model, _ in self.get_model_filename_list():
                if type(model) == keras.models.Model:
                    for layer in model.layers:
                        if type(layer) == FUNITAdain:
                            weights_list += [ x for x in layer.weights if 'kernel' in x.name ]
                        elif  type(layer) == keras.layers.Conv2D or type(layer) == keras.layers.Dense:
                            weights_list += [ layer.weights[0] ]

            initer = keras.initializers.he_normal()
            for w in weights_list:
                K.set_value( w, K.get_value(initer(K.int_shape(w)))  )


        if load_weights_locally:
            pass
        #f weights_file_root is not None:
        #   weights_file_root = Path(weights_file_root)
        #lse:
        #   weights_file_root = Path(__file__).parent
        #elf.weights_path = weights_file_root / ('FUNIT_%s.h5' % (face_type_str) )
        #f load_weights:
        #   self.model.load_weights (str(self.weights_path))



    def get_model_filename_list(self):
        return [[self.enc_class_model, 'enc_class_model.h5'],
                [self.enc_content,     'enc_content.h5'],
                [self.decoder,         'decoder.h5'],
                [self.dis,             'dis.h5'],
                [self.G_opt,           'G_opt.h5'],
                [self.D_opt,           'D_opt.h5'],
                ]

    def train(self, xa,la,xb,lb):
        D_loss, = self.D_train ([xa,la,xb,lb])
        G_loss, = self.G_train ([xa,la,xb,lb])
        return G_loss, D_loss

    def get_average_class_code(self, *args, **kwargs):
        return self.get_average_class_code(*args, **kwargs)

    def convert(self, *args, **kwargs):
        return self.G_convert(*args, **kwargs)

    @staticmethod
    def ContentEncoderFlow(downs=2, nf=64, n_res_blks=2):
        exec (nnlib.import_all(), locals(), globals())

        def ResBlock(dim):
            def func(input):
                x = input
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = InstanceNormalization()(x)
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = InstanceNormalization()(x)

                return Add()([x,input])
            return func

        def func(x):
            x = Conv2D (nf, kernel_size=7, strides=1, padding='same')(x)
            x = InstanceNormalization()(x)
            x = ReLU()(x)
            for i in range(downs):
                x = Conv2D (nf * 2**(i+1), kernel_size=4, strides=2, padding='valid')(ZeroPadding2D(1)(x))
                x = InstanceNormalization()(x)
                x = ReLU()(x)
            for i in range(n_res_blks):
                x = ResBlock( nf * 2**downs )(x)
            return x

        return func

    @staticmethod
    def ClassModelEncoderFlow(downs=4, nf=64, latent_dim=64):
        exec (nnlib.import_all(), locals(), globals())

        def func(x):
            x = Conv2D (nf, kernel_size=7, strides=1, padding='same', activation='relu')(x)
            for i in range(downs):
                x = Conv2D (nf * min ( 4, 2**(i+1) ), kernel_size=4, strides=2, padding='valid', activation='relu')(ZeroPadding2D(1)(x))
            x = GlobalAveragePooling2D()(x)
            x = Dense(latent_dim)(x)
            return x

        return func

    @staticmethod
    def DecoderFlow(ups, n_res_blks=2, mlp_blks=2, subpixel_decoder=False ):
        exec (nnlib.import_all(), locals(), globals())

        def ResBlock(dim):
            def func(input):
                inp, mlp = input
                x = inp
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = FUNITAdain(kernel_initializer='he_normal')([x,mlp])
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = FUNITAdain(kernel_initializer='he_normal')([x,mlp])
                return Add()([x,inp])
            return func

        def func(inputs):
            x , class_code = inputs

            nf = K.int_shape(x)[-1]

            ### MLP block inside decoder
            mlp = class_code
            for i in range(mlp_blks):
                mlp = Dense(nf, activation='relu')(mlp)

            for i in range(n_res_blks):
                x = ResBlock(nf)( [x,mlp] )

            for i in range(ups):

                if subpixel_decoder:
                    x = Conv2D (4* (nf // 2**(i+1)), kernel_size=3, strides=1, padding='same')(x)
                    x = SubpixelUpscaler()(x)
                else:
                    x = UpSampling2D()(x)
                    x = Conv2D (nf // 2**(i+1), kernel_size=5, strides=1, padding='same')(x)

                x = InstanceNormalization()(x)
                x = ReLU()(x)

            rgb = Conv2D (3, kernel_size=7, strides=1, padding='same', activation='tanh')(x)
            return rgb

        return func



    @staticmethod
    def DiscriminatorFlow(nf, n_res_blks, num_classes ):
        exec (nnlib.import_all(), locals(), globals())

        n_layers = n_res_blks // 2

        def ActFirstResBlock(fout):
            def func(x):
                fin = K.int_shape(x)[-1]
                fhid = min(fin, fout)

                if fin != fout:
                    x_s = Conv2D (fout, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
                else:
                    x_s = x

                x = LeakyReLU(0.2)(x)
                x = Conv2D (fhid, kernel_size=3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                x = LeakyReLU(0.2)(x)
                x = Conv2D (fout, kernel_size=3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                return  Add()([x_s, x])

            return func

        def func( x ):
            l_nf = nf
            x = Conv2D (l_nf, kernel_size=7, strides=1, padding='valid')(ZeroPadding2D(3)(x))
            for i in range(n_layers-1):
                l_nf_out = min( l_nf*2, 1024 )
                x = ActFirstResBlock(l_nf)(x)
                x = ActFirstResBlock(l_nf_out)(x)
                x = AveragePooling2D( pool_size=3, strides=2, padding='valid' )(ZeroPadding2D(1)(x))
                l_nf = min( l_nf*2, 1024 )

            l_nf_out = min( l_nf*2, 1024 )
            x        = ActFirstResBlock(l_nf)(x)
            feat = x = ActFirstResBlock(l_nf_out)(x)

            x = LeakyReLU(0.2)(x)
            x = Conv2D (num_classes, kernel_size=1, strides=1, padding='valid')(x)

            return x, feat

        return func