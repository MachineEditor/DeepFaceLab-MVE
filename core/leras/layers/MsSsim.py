from core.leras import nn
tf = nn.tf

class MsSsim(nn.LayerBase):
    default_power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    def __init__(self, resolution, kernel_size=11, **kwargs):
        # restrict mssim factors to those greater/equal to kernel size
        power_factors = [p for i, p in enumerate(self.default_power_factors) if resolution//(2**i) >= kernel_size]
        # normalize power factors if reduced because of size
        if sum(power_factors) < 1.0:
            power_factors = [x/sum(power_factors) for x in power_factors]
        self.power_factors = power_factors
        self.kernel_size = kernel_size

        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, max_val):
        # Transpose images from NCHW to NHWC
        y_true_t = tf.transpose(tf.cast(y_true, tf.float32), [0, 2, 3, 1])
        y_pred_t = tf.transpose(tf.cast(y_pred, tf.float32), [0, 2, 3, 1])


        def assign_device(op):
            if op.type != 'Assert' or op.type != 'ListDiff':
                return '/gpu:0'
            else:
                return '/cpu:0'

        with tf.device(assign_device):
            loss = tf.image.ssim_multiscale(y_true_t, y_pred_t, max_val, power_factors=self.power_factors)
            return (1.0 - loss) / 2.0

nn.MsSsim = MsSsim
