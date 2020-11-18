from tensorflow.python.ops import control_flow_ops, state_ops
from core.leras import nn
tf = nn.tf

class RMSprop(nn.OptimizerBase):
    def __init__(self, lr=0.001, rho=0.9, lr_dropout=1.0, epsilon=1e-7, clipnorm=0.0, name=None):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr_dropout = lr_dropout
        self.clipnorm = clipnorm

        with tf.device('/CPU:0') :
            with tf.variable_scope(self.name):
                self.lr = tf.Variable (lr, name="lr")
                self.rho = tf.Variable (rho, name="rho")
                self.epsilon = tf.Variable (epsilon, name="epsilon")
                self.iterations = tf.Variable(0, dtype=tf.int64, name='iters')

        self.accumulators_dict = {}
        self.lr_rnds_dict = {}

    def get_weights(self):
        return [self.iterations] + list(self.accumulators_dict.values())

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        # Initialize here all trainable variables used in training
        e = tf.device('/CPU:0') if vars_on_cpu else None
        if e: e.__enter__()
        with tf.variable_scope(self.name):
            accumulators = { v.name : tf.get_variable ( f'acc_{v.name}'.replace(':','_'), v.shape, dtype=v.dtype, initializer=tf.initializers.constant(0.0), trainable=False) for v in trainable_weights }
            self.accumulators_dict.update ( accumulators)

            if self.lr_dropout != 1.0:
                e = tf.device('/CPU:0') if lr_dropout_on_cpu else None
                if e: e.__enter__()                    
                lr_rnds = [ nn.random_binomial( v.shape, p=self.lr_dropout, dtype=v.dtype) for v in trainable_weights ]
                if e: e.__exit__(None, None, None)                
                self.lr_rnds_dict.update ( { v.name : rnd for v,rnd in zip(trainable_weights,lr_rnds) } )
        if e: e.__exit__(None, None, None)

    def get_update_op(self, grads_vars):
        updates = []

        if self.clipnorm > 0.0:
            norm = tf.sqrt( sum([tf.reduce_sum(tf.square(g)) for g,v in grads_vars]))
        updates += [ state_ops.assign_add( self.iterations, 1) ]
        for i, (g,v) in enumerate(grads_vars):
            if self.clipnorm > 0.0:
                g = self.tf_clip_norm(g, self.clipnorm, norm)

            a = self.accumulators_dict[ v.name ]

            rho = tf.cast(self.rho, a.dtype)
            new_a = rho * a + (1. - rho) * tf.square(g)

            lr = tf.cast(self.lr, a.dtype)
            epsilon = tf.cast(self.epsilon, a.dtype)

            v_diff = - lr * g / (tf.sqrt(new_a) + epsilon)
            if self.lr_dropout != 1.0:
                lr_rnd = self.lr_rnds_dict[v.name]
                v_diff *= lr_rnd
            new_v = v + v_diff

            updates.append (state_ops.assign(a, new_a))
            updates.append (state_ops.assign(v, new_v))

        return control_flow_ops.group ( *updates, name=self.name+'_updates')
nn.RMSprop = RMSprop