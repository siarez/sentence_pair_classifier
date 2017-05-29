import numpy as np
import tensorflow as tf
import time

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

#a, b are sentence word embeddings
a = tf.placeholder(dtype='float', shape=[None, 300], name='ai')
b = tf.placeholder(dtype='float', shape=[None, 300], name='bi')
label = tf.placeholder(dtype='float', shape=[1], name='label')


Fa = tf.layers.dense(inputs=a, units=200, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='F')

Fb = tf.layers.dense(inputs=b, units=200, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='F', reuse=True)

attention_weights = tf.matmul(Fa, Fb, transpose_b=True, name='attention_weights')

#e_rs = tf.reduce_sum(tf.exp(attention_weights), axis=1, name='sum-eik')

attention_soft1 = tf.nn.softmax(attention_weights, name='attention_soft')
attention_soft2 = tf.nn.softmax(tf.transpose(attention_weights), name='attention_soft')

beta = tf.matmul(attention_soft1, b)
alpha = tf.matmul(attention_soft2, a)

a_beta = tf.concat([a, beta], 1)
b_alpha = tf.concat([b, alpha], 1)

v1i = tf.layers.dense(inputs=a_beta, units=200, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='G')

v2j = tf.layers.dense(inputs=b_alpha, units=200, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='G', reuse=True)

v1 = tf.reduce_sum(v1i, axis=0)
v2 = tf.reduce_sum(v2j, axis=0)

v1_v2 = tf.concat([v1, v2], 0)
v1_v2_1 = tf.reshape(v1_v2, [1, 400])

y = tf.layers.dense(inputs=v1_v2_1, units=1, activation=tf.nn.sigmoid, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='H')

#onehot_labels = tf.one_hot(indices=tf.cast(label, tf.int32), depth=1)
onehot_labels = tf.reshape(label, [1, 1])
#loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=y)

loss = tf.square(tf.reduce_sum(y - onehot_labels))

train_op = tf.contrib.layers.optimize_loss(
    loss=loss,
    global_step=tf.contrib.framework.get_global_step(),
    learning_rate=0.01,
    optimizer="SGD")


init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)

    a_feed = np.random.rand(3, 300)
    b_feed = np.random.rand(5, 300)
    label_feed = np.array([0])

    start_time = time.time()


    for i in range(10000):
        train_op1, losss, y2, one_hots = sess.run([train_op, loss, y, onehot_labels], {a: a_feed, b: b_feed, label: label_feed})
        if i % 100 == 0:
            print(losss, y2, one_hots, 'sps: ', 100/(time.time() - start_time))
            start_time = time.time()
