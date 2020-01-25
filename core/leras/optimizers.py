import copy

def initialize_optimizers(nn):
    tf = nn.tf
    from tensorflow.python.ops import state_ops, control_flow_ops

    class TFBaseOptimizer(nn.Saveable):
        def __init__(self, name=None):
            super().__init__(name=name)

        def tf_clip_norm(self, g, c, n):
            """Clip the gradient `g` if the L2 norm `n` exceeds `c`.
            # Arguments
                g: Tensor, the gradient tensor
                c: float >= 0. Gradients will be clipped
                    when their L2 norm exceeds this value.
                n: Tensor, actual norm of `g`.
            # Returns
                Tensor, the gradient clipped if required.
            """
            if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
                return g

            condition = n >= c
            then_expression = tf.scalar_mul(c / n, g)
            else_expression = g

            # saving the shape to avoid converting sparse tensor to dense
            if isinstance(then_expression, tf.Tensor):
                g_shape = copy.copy(then_expression.get_shape())
            elif isinstance(then_expression, tf.IndexedSlices):
                g_shape = copy.copy(then_expression.dense_shape)
            if condition.dtype != tf.bool:
                condition = tf.cast(condition, 'bool')
            g = tf.cond(condition,
                        lambda: then_expression,
                        lambda: else_expression)
            if isinstance(then_expression, tf.Tensor):
                g.set_shape(g_shape)
            elif isinstance(then_expression, tf.IndexedSlices):
                g._dense_shape = g_shape

            return g
    nn.TFBaseOptimizer = TFBaseOptimizer

    class TFRMSpropOptimizer(TFBaseOptimizer):
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

            self.accumulators = []
            self.accumulator_counter = 0
            self.accumulators_dict = {}
            self.lr_rnds_dict = {}

        def get_weights(self):
            return [self.lr, self.rho, self.epsilon, self.iterations] + self.accumulators

        def initialize_variables(self, trainable_weights, vars_on_cpu=True):
            # Initialize here all trainable variables used in training
            e = tf.device('/CPU:0') if vars_on_cpu else None
            if e: e.__enter__()
            with tf.variable_scope(self.name):
                accumulators = [ tf.get_variable ( f'acc_{i+self.accumulator_counter}', v.shape, dtype=v.dtype, initializer=tf.initializers.constant(0.0), trainable=False)
                                    for (i, v ) in enumerate(trainable_weights) ]

                self.accumulators_dict.update ( { v.name : acc for v,acc in zip(trainable_weights,accumulators) } )
                self.accumulators += accumulators
                self.accumulator_counter += len(trainable_weights)

                if self.lr_dropout != 1.0:
                    lr_rnds = [ nn.tf_random_binomial( v.shape, p=self.lr_dropout, dtype=v.dtype) for v in trainable_weights ]
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

                a = self.accumulators_dict[v.name]

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
    nn.TFRMSpropOptimizer = TFRMSpropOptimizer