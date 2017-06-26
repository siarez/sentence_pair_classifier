import tensorflow as tf

class Model(object):
    """This class constructs the Tensorflow model graph."""
    def __init__(self, mode, name):
        """Init take to arguments. mode to indicate training vs. inference, and name """
        with tf.variable_scope(name):
            assert type(name) is str
            self.mode = mode
            # a, b are sentences with word vectors of length 300
            self.a = tf.placeholder(dtype='float', shape=[None, 300], name='a')
            self.b = tf.placeholder(dtype='float', shape=[None, 300], name='b')
            self.label = tf.placeholder(dtype='float', shape=[1], name='label')

            # Variables to accumulate accuracy measurements
            self.accuracy = tf.Variable(tf.constant(0.0), name='accuracy')
            self.validation_iter = tf.Variable(tf.constant(1), name='validation_iter')

            F1a = tf.layers.dense(inputs=self.a, units=300, activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=None, bias_initializer=None,
                                  name='F1')
            F1b = tf.layers.dense(inputs=self.b, units=300, activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=None, bias_initializer=None,
                                  name='F1', reuse=True)

            Fa = tf.layers.dense(inputs=F1a, units=300, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=None, bias_initializer=None,
                                 name='F2')
            Fb = tf.layers.dense(inputs=F1b, units=300, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=None, bias_initializer=None,
                                 name='F2', reuse=True)

            attention_weights = tf.matmul(Fa, Fb, transpose_b=True, name='attention_weights')
            attention_soft1 = tf.nn.softmax(attention_weights, name='attention_soft1')
            attention_soft2 = tf.nn.softmax(tf.transpose(attention_weights), name='attention_soft2')

            beta = tf.matmul(attention_soft1, self.b)
            alpha = tf.matmul(attention_soft2, self.a)

            a_beta = tf.concat([self.a, beta], 1)
            b_alpha = tf.concat([self.b, alpha], 1)

            v1i1 = tf.layers.dense(inputs=a_beta, units=300, activation=tf.nn.relu, use_bias=True,
                                   kernel_initializer=None, bias_initializer=None,
                                   name='G1')
            v2j1 = tf.layers.dense(inputs=b_alpha, units=300, activation=tf.nn.relu, use_bias=True,
                                   kernel_initializer=None, bias_initializer=None,
                                   name='G1', reuse=True)

            v1i = tf.layers.dense(inputs=v1i1, units=300, activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=None, bias_initializer=None,
                                  name='G2')
            v2j = tf.layers.dense(inputs=v2j1, units=300, activation=tf.nn.relu, use_bias=True,
                                  kernel_initializer=None, bias_initializer=None,
                                  name='G2', reuse=True)

            v1 = tf.reduce_sum(v1i, axis=0)
            v2 = tf.reduce_sum(v2j, axis=0)

            v1_v2 = tf.concat([v1, v2], 0)
            v1_v2_1 = tf.reshape(v1_v2, [1, 600])

            y1 = tf.layers.dense(inputs=v1_v2_1, units=600, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=None, bias_initializer=None,
                                 name='H1')
            self.logits = tf.layers.dense(inputs=y1, units=2, activation=None, use_bias=True,
                                          kernel_initializer=None, bias_initializer=None,
                                          name='H')

            self.loss = None
            self.train_op = None
            self.probabilities = tf.nn.softmax(self.logits, name='softmax_tensor')

            # Calculate Loss (for both TRAIN and EVAL modes)
            if self.mode != 'INFER':
                onehot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=2)
                # self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.logits)
                self.loss = tf.losses.log_loss(onehot_labels, self.probabilities)
                self.loss_summary = tf.summary.scalar('loss', self.loss)

            # Configure the Training Op (for TRAIN mode)
            if self.mode == 'TRAIN':
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=self.loss,
                    global_step=tf.contrib.framework.get_global_step(),
                    learning_rate=0.001, optimizer='SGD')

            self.classes = tf.argmax(input=self.logits, axis=1, name='classes'),

            # self.accuracy variable is used for validation. It holds and average of correct classifications.
            # At the start of each validation cycle self.accuracy and self.validation_iter are reset.
            # This average is updated after each sample is classified. Below is a more readable form:
            # accuracy := (accuracy * (validation_iter - 1) + (prediction == label)) / validation_iter
            self.accuracy_op = self.accuracy.assign(
                (self.accuracy * (tf.cast(self.validation_iter, dtype=tf.float32) - tf.constant(1.0)) +
                 (tf.reduce_sum(tf.cast(tf.equal(self.classes, tf.cast(self.label, tf.int64)), dtype=tf.float32)))) /
                tf.cast(self.validation_iter, dtype=tf.float32))

            self.accuracy_summary_op = tf.summary.scalar('accuracy', self.accuracy)
            self.validation_iter_op = self.validation_iter.assign_add(1)

