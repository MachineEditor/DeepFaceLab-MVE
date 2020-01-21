import numpy as np

def initialize_tensor_ops(nn):
    tf = nn.tf
    from tensorflow.python.ops import array_ops, random_ops, math_ops, sparse_ops, gradients      
    from tensorflow.python.framework import sparse_tensor
    
    def tf_get_value(tensor):
        return nn.tf_sess.run (tensor)
    nn.tf_get_value = tf_get_value
    
    
    def tf_batch_set_value(tuples):
        if len(tuples) != 0:
            with nn.tf.device('/CPU:0'):
                assign_ops = []
                feed_dict = {}

                for x, value in tuples:
                    if isinstance(value, nn.tf.Operation):
                        assign_ops.append(value)
                    else:
                        value = np.asarray(value, dtype=x.dtype.as_numpy_dtype)
                        assign_placeholder = nn.tf.placeholder( x.dtype.base_dtype, shape=[None]*value.ndim )
                        assign_op = nn.tf.assign (x, assign_placeholder )
                        assign_ops.append(assign_op)
                        feed_dict[assign_placeholder] = value

                nn.tf_sess.run(assign_ops, feed_dict=feed_dict)
    nn.tf_batch_set_value = tf_batch_set_value
    
    
    def tf_gradients ( loss, vars ):
        grads = gradients.gradients(loss, vars, colocate_gradients_with_ops=True )
        #todo none gradient for var
        return [*zip(grads,vars)]
    nn.tf_gradients = tf_gradients
    
    def tf_average_gv_list(grad_var_list, tf_device_string=None):
        e = tf.device(tf_device_string) if tf_device_string is not None else None
        if e is not None: e.__enter__()
        result = []
        for i, (gv) in enumerate(grad_var_list):
            for j,(g,v) in enumerate(gv):
                g = tf.expand_dims(g, 0)
                if i == 0:
                    result += [ [[g], v]  ]
                else:
                    result[j][0] += [g]

        for i,(gs,v) in enumerate(result):
            result[i] = ( tf.reduce_mean( tf.concat (gs, 0), 0 ), v )
        if e is not None: e.__exit__(None,None,None)
        return result
    nn.tf_average_gv_list = tf_average_gv_list
    
    def tf_average_tensor_list(tensors_list, tf_device_string=None):
        e = tf.device(tf_device_string) if tf_device_string is not None else None
        if e is not None: e.__enter__()
        result = tf.reduce_mean(tf.concat ([tf.expand_dims(t, 0) for t in tensors_list], 0), 0)
        if e is not None: e.__exit__(None,None,None)
        return result
    nn.tf_average_tensor_list = tf_average_tensor_list
    
    def tf_dot(x, y):
        if x.shape.ndims > 2 or y.shape.ndims > 2:
            x_shape = []
            for i, s in zip( x.shape.as_list(), array_ops.unstack(array_ops.shape(x))):
                if i is not None:
                    x_shape.append(i)
                else:
                    x_shape.append(s)
            x_shape = tuple(x_shape)
            y_shape = []
            for i, s in zip( y.shape.as_list(), array_ops.unstack(array_ops.shape(y))):
                if i is not None:
                    y_shape.append(i)
                else:
                    y_shape.append(s)
            y_shape = tuple(y_shape)
            y_permute_dim = list(range(y.shape.ndims))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = array_ops.reshape(x, [-1, x_shape[-1]])
            yt = array_ops.reshape(array_ops.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
            
            import code
            code.interact(local=dict(globals(), **locals()))
            return array_ops.reshape(math_ops.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
        if isinstance(x, sparse_tensor.SparseTensor):
            out = sparse_ops.sparse_tensor_dense_matmul(x, y)
        else:
            out = math_ops.matmul(x, y)
        return out
    nn.tf_dot = tf_dot
    
    def tf_gelu(x):
        cdf = 0.5 * (1.0 + tf.nn.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf
    nn.tf_gelu = tf_gelu
     
    def tf_upsample2d(x, size=2):
        return tf.image.resize_nearest_neighbor(x, (x.shape[1]*size, x.shape[2]*size) )
    nn.tf_upsample2d = tf_upsample2d
    
    def tf_upsample2d_bilinear(x, size=2):
        return tf.image.resize_images(x, (x.shape[1]*size, x.shape[2]*size) )
    nn.tf_upsample2d_bilinear = tf_upsample2d_bilinear
    
    def tf_flatten(x, dynamic_dims=False):
        """
        dynamic_dims allows to flatten without knowing size on input dims
        """
        if dynamic_dims:
            sh = tf.shape(x)
            return tf.reshape (x, (sh[0], tf.reduce_prod(sh[1:]) ) )
        else:
            return tf.reshape (x, (-1, np.prod(x.shape[1:])) )
        
    nn.tf_flatten = tf_flatten
    
    def tf_random_binomial(shape, p=0.0, dtype=None, seed=None):
        if dtype is None:
            dtype=tf.float32

        if seed is None:
            seed = np.random.randint(10e6)
        return array_ops.where(
            random_ops.random_uniform(shape, dtype=tf.float16, seed=seed) < p,
            array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))
    nn.tf_random_binomial = tf_random_binomial
    
    def tf_gaussian_blur(input, radius=2.0):
        def gaussian(x, mu, sigma):
            return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

        def make_kernel(sigma):
            kernel_size = max(3, int(2 * 2 * sigma + 1))
            mean = np.floor(0.5 * kernel_size)
            kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
            np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
            kernel = np_kernel / np.sum(np_kernel)
            return kernel

        gauss_kernel = make_kernel(radius)
        gauss_kernel = gauss_kernel[:, :,np.newaxis, np.newaxis]
        kernel_size = gauss_kernel.shape[0]

        inputs = [ input[:,:,:,i:i+1]  for i in range( input.shape[-1] ) ]

        outputs = []
        for i in range(len(inputs)):
            x = inputs[i]
            if kernel_size != 0:
                padding = kernel_size//2
                x = tf.pad (x, [ [0,0], [padding,padding], [padding,padding], [0,0] ] )

            outputs += [ tf.nn.conv2d(x, tf.constant(gauss_kernel, dtype=nn.tf_floatx ) , strides=[1,1,1,1], padding="VALID") ]

        return tf.concat (outputs, axis=-1)
    nn.tf_gaussian_blur = tf_gaussian_blur
    
    def tf_style_loss(target, style, gaussian_blur_radius=0.0, loss_weight=1.0, step_size=1):
        def sd(content, style, loss_weight):
            content_nc = content.shape[-1]
            style_nc = style.shape[-1]
            if content_nc != style_nc:
                raise Exception("style_loss() content_nc != style_nc")

            axes = [1,2]
            c_mean, c_var = tf.nn.moments(content, axes=axes, keep_dims=True)
            s_mean, s_var = tf.nn.moments(style, axes=axes, keep_dims=True)
            c_std, s_std = tf.sqrt(c_var + 1e-5), tf.sqrt(s_var + 1e-5)

            mean_loss = tf.reduce_sum(tf.square(c_mean-s_mean), axis=[1,2,3])
            std_loss  = tf.reduce_sum(tf.square(c_std-s_std), axis=[1,2,3])

            return (mean_loss + std_loss) * ( loss_weight / content_nc.value )

        if gaussian_blur_radius > 0.0:
            target = tf_gaussian_blur(target, gaussian_blur_radius)
            style = tf_gaussian_blur(style, gaussian_blur_radius)

        return sd( target, style, loss_weight=loss_weight )

    nn.tf_style_loss = tf_style_loss
    
    def tf_dssim(img1,img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    
        ch = img2.shape[-1]

        def _fspecial_gauss(size, sigma):
            #Function to mimic the 'fspecial' gaussian MATLAB function.
            coords = np.arange(0, size, dtype=nn.np_floatx)
            coords -= (size - 1 ) / 2.0
            g = coords**2
            g *= ( -0.5 / (sigma**2) )
            g = np.reshape (g, (1,-1)) + np.reshape(g, (-1,1) )
            g = tf.constant ( np.reshape (g, (1,-1)), dtype=nn.tf_floatx )
            g = tf.nn.softmax(g)
            g = tf.reshape (g, (size, size, 1, 1))
            g = tf.tile (g, (1,1,ch,1))
            return g

        kernel = _fspecial_gauss(filter_size,filter_sigma)

        def reducer(x):
            return tf.nn.depthwise_conv2d(x, kernel, strides=[1,1,1,1], padding='VALID')

        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2

        mean0 = reducer(img1)
        mean1 = reducer(img2)
        num0 = mean0 * mean1 * 2.0
        den0 = tf.square(mean0) + tf.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)

        num1 = reducer(img1 * img2) * 2.0
        den1 = reducer(tf.square(img1) + tf.square(img2))
        c2 *= 1.0 #compensation factor
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        ssim_val = tf.reduce_mean(luminance * cs, axis=(-3, -2) )
        return(1.0 - ssim_val ) / 2.0
    nn.tf_dssim = tf_dssim
    
    def tf_rgb_to_lab(srgb):
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
        exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        rgb_to_xyz = tf.constant([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334], # R
            [0.357580, 0.715160, 0.119193], # G
            [0.180423, 0.072169, 0.950227], # B
        ])
        xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

        epsilon = 6/29
        linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
        exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

        fxfyfz_to_lab = tf.constant([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
        ])
        lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
        return tf.reshape(lab_pixels, tf.shape(srgb))
    nn.tf_rgb_to_lab = tf_rgb_to_lab
    
    def tf_suppress_lower_mean(t, eps=0.00001):                        
        if t.shape.ndims != 1:
            raise ValueError("tf_suppress_lower_mean: t rank must be 1")        
        t_mean_eps = tf.reduce_mean(t) - eps                    
        q = tf.clip_by_value(t, t_mean_eps, tf.reduce_max(t) )   
        q = tf.clip_by_value(q-t_mean_eps, 0, eps)
        q = q * (t/eps)                         
        return q
"""
class GeLU(KL.Layer):
            Gaussian Error Linear Unit.
            A smoother version of ReLU generally used
            in the BERT or BERT architecture based models.
            Original paper: https://arxiv.org/abs/1606.08415
            Input shape:
                Arbitrary. Use the keyword argument `input_shape`
                (tuple of integers, does not include the samples axis)
                when using this layer as the first layer in a model.
            Output shape:
                Same shape as the input.

            def __init__(self, approximate=True, **kwargs):
                super(GeLU, self).__init__(**kwargs)
                self.approximate = approximate
                self.supports_masking = True

            def call(self, inputs):
                cdf = 0.5 * (1.0 + K.tanh((np.sqrt(2 / np.pi) * (inputs + 0.044715 * K.pow(inputs, 3)))))
                return inputs * cdf

            def get_config(self):
                config = {'approximate': self.approximate}
                base_config = super(GeLU, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            def compute_output_shape(self, input_shape):
                return input_shape
        nn.GeLU = GeLU
"""