import pickle
from pathlib import Path
from core import pathex
from core.interact import interact as io
import numpy as np


def initialize_layers(nn):
    tf = nn.tf

    class Saveable():
        def __init__(self, name=None):
            self.name = name

        #override
        def get_weights(self):
            #return tf tensors that should be initialized/loaded/saved
            pass

        def save_weights(self, filename, force_dtype=None):
            d = {}
            weights = self.get_weights()

            if self.name is None:
                raise Exception("name must be defined.")

            name = self.name
            for w, w_val in zip(weights, nn.tf_sess.run (weights)):
                w_name_split = w.name.split('/', 1)
                if name != w_name_split[0]:
                    raise Exception("weight first name != Saveable.name")

                if force_dtype is not None:
                    w_val = w_val.astype(force_dtype)

                d[ w_name_split[1] ] = w_val

            d_dumped = pickle.dumps (d, 4)
            pathex.write_bytes_safe ( Path(filename), d_dumped )

        def load_weights(self, filename):
            """
            returns True if file exists
            """
            filepath = Path(filename)
            if filepath.exists():
                result = True
                d_dumped = filepath.read_bytes()
                d = pickle.loads(d_dumped)
            else:
                return False

            weights = self.get_weights()

            if self.name is None:
                raise Exception("name must be defined.")

            tuples = []
            for w in weights:
                w_name_split = w.name.split('/')
                if self.name != w_name_split[0]:
                    raise Exception("weight first name != Saveable.name")

                sub_w_name = "/".join(w_name_split[1:])

                w_val = d.get(sub_w_name, None)
                w_val = np.reshape( w_val, w.shape.as_list() )

                if w_val is None:
                    io.log_err(f"Weight {w.name} was not loaded from file {filename}")
                    tuples.append ( (w, w.initializer) )
                else:
                    tuples.append ( (w, w_val) )

            nn.tf_batch_set_value(tuples)

            return True

        def init_weights(self):
            nn.tf_init_weights(self.get_weights())
    nn.Saveable = Saveable

    class LayerBase():
        def __init__(self, name=None, **kwargs):
            self.name = name

        #override
        def build_weights(self):
            pass

        #override
        def get_weights(self):
            return []

        def set_weights(self, new_weights):
            weights = self.get_weights()
            if len(weights) != len(new_weights):
                raise ValueError ('len of lists mismatch')

            tuples = []
            for w, new_w in zip(weights, new_weights):
                if len(w.shape) != new_w.shape:
                    new_w = new_w.reshape(w.shape)

                tuples.append ( (w, new_w) )

            nn.tf_batch_set_value (tuples)
    nn.LayerBase = LayerBase

    class Conv2D(LayerBase):
        """
        use_wscale  bool enables equalized learning rate, kernel_initializer will be forced to random_normal


        """
        def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
            if not isinstance(strides, int):
                raise ValueError ("strides must be an int type")
            if not isinstance(dilations, int):
                raise ValueError ("dilations must be an int type")
            kernel_size = int(kernel_size)

            if dtype is None:
                dtype = nn.tf_floatx

            if isinstance(padding, str):
                if padding == "SAME":
                    padding = ( (kernel_size - 1) * dilations + 1 ) // 2
                elif padding == "VALID":
                    padding = 0
                else:
                    raise ValueError ("Wrong padding type. Should be VALID SAME or INT or 4x INTs")

            if isinstance(padding, int):
                if padding != 0:
                    if nn.data_format == "NHWC":
                        padding = [ [0,0], [padding,padding], [padding,padding], [0,0] ]
                    else:
                        padding = [ [0,0], [0,0], [padding,padding], [padding,padding] ]
                else:
                    padding = None

            if nn.data_format == "NHWC":
                strides = [1,strides,strides,1]
            else:
                strides = [1,1,strides,strides]

            if nn.data_format == "NHWC":
                dilations = [1,dilations,dilations,1]
            else:
                dilations = [1,1,dilations,dilations]

            self.in_ch = in_ch
            self.out_ch = out_ch
            self.kernel_size = kernel_size
            self.strides = strides
            self.padding = padding
            self.dilations = dilations
            self.use_bias = use_bias
            self.use_wscale = use_wscale
            self.kernel_initializer = kernel_initializer
            self.bias_initializer = bias_initializer
            self.trainable = trainable
            self.dtype = dtype
            super().__init__(**kwargs)

        def build_weights(self):
            kernel_initializer = self.kernel_initializer
            if self.use_wscale:
                gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
                fan_in = self.kernel_size*self.kernel_size*self.in_ch
                he_std = gain / np.sqrt(fan_in) # He init
                self.wscale = tf.constant(he_std, dtype=self.dtype )
                kernel_initializer = tf.initializers.random_normal(0, 1.0, dtype=self.dtype)

            if kernel_initializer is None:
                kernel_initializer = tf.initializers.glorot_uniform(dtype=self.dtype)

            self.weight = tf.get_variable("weight", (self.kernel_size,self.kernel_size,self.in_ch,self.out_ch), dtype=self.dtype, initializer=kernel_initializer, trainable=self.trainable )

            if self.use_bias:
                bias_initializer = self.bias_initializer
                if bias_initializer is None:
                    bias_initializer = tf.initializers.zeros(dtype=self.dtype)

                self.bias = tf.get_variable("bias", (self.out_ch,), dtype=self.dtype, initializer=bias_initializer, trainable=self.trainable )

        def get_weights(self):
            weights = [self.weight]
            if self.use_bias:
                weights += [self.bias]
            return weights

        def __call__(self, x):
            weight = self.weight
            if self.use_wscale:
                weight = weight * self.wscale

            if self.padding is not None:
                x = tf.pad (x, self.padding, mode='CONSTANT')

            x = tf.nn.conv2d(x, weight, self.strides, 'VALID', dilations=self.dilations, data_format=nn.data_format)
            if self.use_bias:
                if nn.data_format == "NHWC":
                    bias = tf.reshape (self.bias, (1,1,1,self.out_ch) )
                else:
                    bias = tf.reshape (self.bias, (1,self.out_ch,1,1) )
                x = tf.add(x, bias)
            return x

        def __str__(self):
            r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "

            return r
    nn.Conv2D = Conv2D

    class Conv2DTranspose(LayerBase):
        """
        use_wscale      enables weight scale (equalized learning rate)
                        kernel_initializer will be forced to random_normal
        """
        def __init__(self, in_ch, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
            if not isinstance(strides, int):
                raise ValueError ("strides must be an int type")
            kernel_size = int(kernel_size)

            if dtype is None:
                dtype = nn.tf_floatx

            self.in_ch = in_ch
            self.out_ch = out_ch
            self.kernel_size = kernel_size
            self.strides = strides
            self.padding = padding
            self.use_bias = use_bias
            self.use_wscale = use_wscale
            self.kernel_initializer = kernel_initializer
            self.bias_initializer = bias_initializer
            self.trainable = trainable
            self.dtype = dtype
            super().__init__(**kwargs)

        def build_weights(self):
            kernel_initializer = self.kernel_initializer
            if self.use_wscale:
                gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
                fan_in = self.kernel_size*self.kernel_size*self.in_ch
                he_std = gain / np.sqrt(fan_in) # He init
                self.wscale = tf.constant(he_std, dtype=self.dtype )
                kernel_initializer = tf.initializers.random_normal(0, 1.0, dtype=self.dtype)
            if kernel_initializer is None:
                kernel_initializer = tf.initializers.glorot_uniform(dtype=self.dtype)
            self.weight = tf.get_variable("weight", (self.kernel_size,self.kernel_size,self.out_ch,self.in_ch), dtype=self.dtype, initializer=kernel_initializer, trainable=self.trainable )

            if self.use_bias:
                bias_initializer = self.bias_initializer
                if bias_initializer is None:
                    bias_initializer = tf.initializers.zeros(dtype=self.dtype)

                self.bias = tf.get_variable("bias", (self.out_ch,), dtype=self.dtype, initializer=bias_initializer, trainable=self.trainable )

        def get_weights(self):
            weights = [self.weight]
            if self.use_bias:
                weights += [self.bias]
            return weights

        def __call__(self, x):
            shape = x.shape

            if nn.data_format == "NHWC":
                h,w,c = shape[1], shape[2], shape[3]
                output_shape = tf.stack ( (tf.shape(x)[0],
                                        self.deconv_length(w, self.strides, self.kernel_size, self.padding),
                                        self.deconv_length(h, self.strides, self.kernel_size, self.padding),
                                        self.out_ch) )

                strides = [1,self.strides,self.strides,1]
            else:
                c,h,w = shape[1], shape[2], shape[3]
                output_shape = tf.stack ( (tf.shape(x)[0],
                                           self.out_ch,
                                           self.deconv_length(w, self.strides, self.kernel_size, self.padding),
                                           self.deconv_length(h, self.strides, self.kernel_size, self.padding),
                                           ) )
                strides = [1,1,self.strides,self.strides]
            weight = self.weight
            if self.use_wscale:
                weight = weight * self.wscale

            x = tf.nn.conv2d_transpose(x, weight, output_shape, strides, padding=self.padding, data_format=nn.data_format)

            if self.use_bias:
                if nn.data_format == "NHWC":
                    bias = tf.reshape (self.bias, (1,1,1,self.out_ch) )
                else:
                    bias = tf.reshape (self.bias, (1,self.out_ch,1,1) )
                x = tf.add(x, bias)
            return x

        def __str__(self):
            r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "

            return r

        def deconv_length(self, dim_size, stride_size, kernel_size, padding):
            assert padding in {'SAME', 'VALID', 'FULL'}
            if dim_size is None:
                return None
            if padding == 'VALID':
                dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
            elif padding == 'FULL':
                dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
            elif padding == 'SAME':
                dim_size = dim_size * stride_size
            return dim_size
    nn.Conv2DTranspose = Conv2DTranspose

    class BlurPool(LayerBase):
        def __init__(self, filt_size=3, stride=2, **kwargs ):

            if nn.data_format == "NHWC":
                self.strides = [1,stride,stride,1]
            else:
                self.strides = [1,1,stride,stride]

            self.filt_size = filt_size
            pad = [ int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ]

            if nn.data_format == "NHWC":
                self.padding = [ [0,0], pad, pad, [0,0] ]
            else:
                self.padding = [ [0,0], [0,0], pad, pad ]

            if(self.filt_size==1):
                a = np.array([1.,])
            elif(self.filt_size==2):
                a = np.array([1., 1.])
            elif(self.filt_size==3):
                a = np.array([1., 2., 1.])
            elif(self.filt_size==4):
                a = np.array([1., 3., 3., 1.])
            elif(self.filt_size==5):
                a = np.array([1., 4., 6., 4., 1.])
            elif(self.filt_size==6):
                a = np.array([1., 5., 10., 10., 5., 1.])
            elif(self.filt_size==7):
                a = np.array([1., 6., 15., 20., 15., 6., 1.])

            a = a[:,None]*a[None,:]
            a = a / np.sum(a)
            a = a[:,:,None,None]
            self.a = a
            super().__init__(**kwargs)

        def build_weights(self):
            self.k = tf.constant (self.a, dtype=nn.tf_floatx )

        def __call__(self, x):
            k = tf.tile (self.k, (1,1,x.shape[nn.conv2d_ch_axis],1) )
            x = tf.pad(x, self.padding )
            x = tf.nn.depthwise_conv2d(x, k, self.strides, 'VALID', data_format=nn.data_format)
            return x
    nn.BlurPool = BlurPool

    class Dense(LayerBase):
        def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
            """
            use_wscale          enables weight scale (equalized learning rate)
                                kernel_initializer will be forced to random_normal

            maxout_ch     https://link.springer.com/article/10.1186/s40537-019-0233-0
                                typical 2-4 if you want to enable DenseMaxout behaviour
            """
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.use_bias = use_bias
            self.use_wscale = use_wscale
            self.maxout_ch = maxout_ch
            self.kernel_initializer = kernel_initializer
            self.bias_initializer = bias_initializer
            self.trainable = trainable
            if dtype is None:
                dtype = nn.tf_floatx

            self.dtype = dtype
            super().__init__(**kwargs)

        def build_weights(self):
            if self.maxout_ch > 1:
                weight_shape = (self.in_ch,self.out_ch*self.maxout_ch)
            else:
                weight_shape = (self.in_ch,self.out_ch)

            kernel_initializer = self.kernel_initializer

            if self.use_wscale:
                gain = 1.0
                fan_in = np.prod( weight_shape[:-1] )
                he_std = gain / np.sqrt(fan_in) # He init
                self.wscale = tf.constant(he_std, dtype=self.dtype )
                kernel_initializer = tf.initializers.random_normal(0, 1.0, dtype=self.dtype)

            if kernel_initializer is None:
                kernel_initializer = tf.initializers.glorot_uniform(dtype=self.dtype)

            self.weight = tf.get_variable("weight", weight_shape, dtype=self.dtype, initializer=kernel_initializer, trainable=self.trainable )

            if self.use_bias:
                bias_initializer = self.bias_initializer
                if bias_initializer is None:
                    bias_initializer = tf.initializers.zeros(dtype=self.dtype)
                self.bias = tf.get_variable("bias", (self.out_ch,), dtype=self.dtype, initializer=bias_initializer, trainable=self.trainable )

        def get_weights(self):
            weights = [self.weight]
            if self.use_bias:
                weights += [self.bias]
            return weights

        def __call__(self, x):
            weight = self.weight
            if self.use_wscale:
                weight = weight * self.wscale

            x = tf.matmul(x, weight)

            if self.maxout_ch > 1:
                x = tf.reshape (x, (-1, self.out_ch, self.maxout_ch) )
                x = tf.reduce_max(x, axis=-1)

            if self.use_bias:
                x = tf.add(x, tf.reshape(self.bias, (1,self.out_ch) ) )

            return x
    nn.Dense = Dense

    class BatchNorm2D(LayerBase):
        """
        currently not for training
        """
        def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, **kwargs):
            self.dim = dim
            self.eps = eps
            self.momentum = momentum
            if dtype is None:
                dtype = nn.tf_floatx
            self.dtype = dtype
            super().__init__(**kwargs)

        def build_weights(self):
            self.weight       = tf.get_variable("weight",   (self.dim,), dtype=self.dtype, initializer=tf.initializers.ones() )
            self.bias         = tf.get_variable("bias",     (self.dim,), dtype=self.dtype, initializer=tf.initializers.zeros() )
            self.running_mean = tf.get_variable("running_mean", (self.dim,), dtype=self.dtype, initializer=tf.initializers.zeros(), trainable=False )
            self.running_var  = tf.get_variable("running_var",  (self.dim,), dtype=self.dtype, initializer=tf.initializers.zeros(), trainable=False )

        def get_weights(self):
            return [self.weight, self.bias, self.running_mean, self.running_var]

        def __call__(self, x):
            if nn.data_format == "NHWC":
                shape = (1,1,1,self.dim)
            else:
                shape = (1,self.dim,1,1)

            weight       = tf.reshape ( self.weight      , shape )
            bias         = tf.reshape ( self.bias        , shape )
            running_mean = tf.reshape ( self.running_mean, shape )
            running_var  = tf.reshape ( self.running_var , shape )

            x = (x - running_mean) / tf.sqrt( running_var + self.eps )
            x *= weight
            x += bias
            return x

    nn.BatchNorm2D = BatchNorm2D