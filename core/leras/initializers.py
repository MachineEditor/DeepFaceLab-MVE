import numpy as np

def initialize_initializers(nn):
    tf = nn.tf
    from tensorflow.python.ops import init_ops
    
    class initializers():
        class ca (init_ops.Initializer):
            def __init__(self, dtype=None):
                pass
            
            def __call__(self, shape, dtype=None, partition_info=None):
                return tf.zeros( shape, name="_cai_")

            @staticmethod
            def generate(shape, eps_std=0.05, dtype=np.float32):
                """
                Super fast implementation of Convolution Aware Initialization for 4D shapes
                Convolution Aware Initialization https://arxiv.org/abs/1702.06295
                """
                if len(shape) != 4:
                    raise ValueError("only shape with rank 4 supported.")

                row, column, stack_size, filters_size = shape

                fan_in = stack_size * (row * column)

                kernel_shape = (row, column)

                kernel_fft_shape = np.fft.rfft2(np.zeros(kernel_shape)).shape

                basis_size = np.prod(kernel_fft_shape)
                if basis_size == 1:
                    x = np.random.normal( 0.0, eps_std, (filters_size, stack_size, basis_size) )
                else:
                    nbb = stack_size // basis_size + 1
                    x = np.random.normal(0.0, 1.0, (filters_size, nbb, basis_size, basis_size))
                    x = x + np.transpose(x, (0,1,3,2) ) * (1-np.eye(basis_size))
                    u, _, v = np.linalg.svd(x)
                    x = np.transpose(u, (0,1,3,2) )
                    x = np.reshape(x, (filters_size, -1, basis_size) )
                    x = x[:,:stack_size,:]

                x = np.reshape(x, ( (filters_size,stack_size,) + kernel_fft_shape ) )

                x = np.fft.irfft2( x, kernel_shape ) \
                    + np.random.normal(0, eps_std, (filters_size,stack_size,)+kernel_shape)

                x = x * np.sqrt( (2/fan_in) / np.var(x) )
                x = np.transpose( x, (2, 3, 1, 0) )
                return x.astype(dtype)
    nn.initializers = initializers