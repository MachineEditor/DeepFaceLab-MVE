import pickle
import types
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

    class ModelBase(Saveable):
        def __init__(self, *args, name=None, **kwargs):
            super().__init__(name=name)
            self.layers = []
            self.built = False
            self.args = args
            self.kwargs = kwargs
            self.run_placeholders = None

        def _build_sub(self, layer, name):
            if isinstance (layer, list):
                for i,sublayer in enumerate(layer):
                    self._build_sub(sublayer, f"{name}_{i}")
            elif isinstance (layer, LayerBase) or \
                    isinstance (layer, ModelBase):

                if layer.name is None:
                    layer.name = name

                if isinstance (layer, LayerBase):
                    with tf.variable_scope(layer.name):
                        layer.build_weights()
                elif isinstance (layer, ModelBase):
                    layer.build()

                self.layers.append (layer)

        def xor_list(self, lst1, lst2):
            return  [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)  ]

        def build(self):
            with tf.variable_scope(self.name):

                current_vars = []
                generator = None
                while True:

                    if generator is None:
                        generator = self.on_build(*self.args, **self.kwargs)
                        if not isinstance(generator, types.GeneratorType):
                            generator = None

                    if generator is not None:
                        try:
                            next(generator)
                        except StopIteration:
                            generator = None

                    v = vars(self)
                    new_vars = self.xor_list (current_vars, list(v.keys()) )

                    for name in new_vars:
                        self._build_sub(v[name],name)

                    current_vars += new_vars

                    if generator is None:
                        break

            self.built = True

        #override
        def get_weights(self):
            if not self.built:
                self.build()

            weights = []
            for layer in self.layers:
                weights += layer.get_weights()
            return weights

        def get_layers(self):
            if not self.built:
                self.build()
            layers = []
            for layer in self.layers:
                if isinstance (layer, LayerBase):
                    layers.append(layer)
                else:
                    layers += layer.get_layers()
            return layers

        #override
        def on_build(self, *args, **kwargs):
            """
            init model layers here

            return 'yield' if build is not finished
                        therefore dependency models will be initialized
            """
            pass

        #override
        def forward(self, *args, **kwargs):
            #flow layers/models/tensors here
            pass

        def __call__(self, *args, **kwargs):
            if not self.built:
                self.build()

            return self.forward(*args, **kwargs)

        def compute_output_shape(self, shapes):
            if not self.built:
                self.build()

            not_list = False
            if not isinstance(shapes, list):
                not_list = True
                shapes = [shapes]

            with tf.device('/CPU:0'):
                # CPU tensors will not impact any performance, only slightly RAM "leakage"
                phs = []
                for dtype,sh in shapes:
                    phs += [ tf.placeholder(dtype, sh) ]

                result = self.__call__(phs[0] if not_list else phs)

                if not isinstance(result, list):
                    result = [result]

                result_shapes = []

                for t in result:
                    result_shapes += [ t.shape.as_list() ]

                return result_shapes[0] if not_list else result_shapes

        def compute_output_channels(self, shapes):
            shape = self.compute_output_shape(shapes)
            shape_len = len(shape)

            if shape_len == 4:
                if nn.data_format == "NCHW":
                    return shape[1]
            return shape[-1]

        def build_for_run(self, shapes_list):
            if not isinstance(shapes_list, list):
                raise ValueError("shapes_list must be a list.")

            self.run_placeholders = []
            for dtype,sh in shapes_list:
                self.run_placeholders.append ( tf.placeholder(dtype, sh) )

            self.run_output = self.__call__(self.run_placeholders)

        def run (self, inputs):
            if self.run_placeholders is None:
                raise Exception ("Model didn't build for run.")

            if len(inputs) != len(self.run_placeholders):
                raise ValueError("len(inputs) != self.run_placeholders")

            feed_dict = {}
            for ph, inp in zip(self.run_placeholders, inputs):
                feed_dict[ph] = inp

            return nn.tf_sess.run ( self.run_output, feed_dict=feed_dict)

        def summary(self):
            layers = self.get_layers()
            layers_names = []
            layers_params = []

            max_len_str = 0
            max_len_param_str = 0
            delim_str = "-"

            total_params = 0

            #Get layers names and str lenght for delim
            for l in layers:
                if len(str(l))>max_len_str:
                    max_len_str = len(str(l))
                layers_names+=[str(l).capitalize()]

            #Get params for each layer
            layers_params = [ int(np.sum(np.prod(w.shape) for w in l.get_weights())) for l in layers ]
            total_params = np.sum(layers_params)

            #Get str lenght for delim
            for p in layers_params:
                if len(str(p))>max_len_param_str:
                    max_len_param_str=len(str(p))

            #Set delim
            for i in range(max_len_str+max_len_param_str+3):
                delim_str += "-"

            output = "\n"+delim_str+"\n"

            #Format model name str
            model_name_str = "| "+self.name.capitalize()
            len_model_name_str = len(model_name_str)
            for i in range(len(delim_str)-len_model_name_str):
                model_name_str+= " " if i!=(len(delim_str)-len_model_name_str-2) else " |"

            output += model_name_str +"\n"
            output += delim_str +"\n"


            #Format layers table
            for i in range(len(layers_names)):
                output += delim_str +"\n"

                l_name = layers_names[i]
                l_param = str(layers_params[i])
                l_param_str = ""
                if len(l_name)<=max_len_str:
                    for i in range(max_len_str - len(l_name)):
                        l_name+= " "

                if len(l_param)<=max_len_param_str:
                    for i in range(max_len_param_str - len(l_param)):
                        l_param_str+= " "

                l_param_str += l_param


                output +="| "+l_name+"|"+l_param_str+"| \n"

            output += delim_str +"\n"

            #Format sum of params
            total_params_str = "| Total params count: "+str(total_params)
            len_total_params_str = len(total_params_str)
            for i in range(len(delim_str)-len_total_params_str):
                total_params_str+= " " if i!=(len(delim_str)-len_total_params_str-2) else " |"

            output += total_params_str +"\n"
            output += delim_str +"\n"

            io.log_info(output)

    nn.ModelBase = ModelBase

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
            self.strides = [1,stride,stride,1]
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
            k = tf.tile (self.k, (1,1,x.shape[-1],1) )
            x = tf.pad(x, self.padding )
            x = tf.nn.depthwise_conv2d(x, k, self.strides, 'VALID')
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