import numpy as np
import tensorflow as tf
import time
import os.path
import pandas as pd
import nltk
from tqdm import tqdm


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


'''    Model definition    '''
#a, b are sentence word embeddings
a = tf.placeholder(dtype='float', shape=[None, 300], name='ai')
b = tf.placeholder(dtype='float', shape=[None, 300], name='bi')
label = tf.placeholder(dtype='float', shape=[1], name='label')


F1a = tf.layers.dense(inputs=a, units=300, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='F1')
F1b = tf.layers.dense(inputs=b, units=300, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='F1', reuse=True)

Fa = tf.layers.dense(inputs=F1a, units=300, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='F2')
Fb = tf.layers.dense(inputs=F1b, units=300, activation=tf.nn.relu, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='F2', reuse=True)



attention_weights = tf.matmul(Fa, Fb, transpose_b=True, name='attention_weights')

#e_rs = tf.reduce_sum(tf.exp(attention_weights), axis=1, name='sum-eik')

attention_soft1 = tf.nn.softmax(attention_weights, name='attention_soft')
attention_soft2 = tf.nn.softmax(tf.transpose(attention_weights), name='attention_soft')

beta = tf.matmul(attention_soft1, b)
alpha = tf.matmul(attention_soft2, a)

a_beta = tf.concat([a, beta], 1)
b_alpha = tf.concat([b, alpha], 1)

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

y = tf.layers.dense(inputs=v1_v2_1, units=1, activation=tf.nn.sigmoid, use_bias=True,
                kernel_initializer=None, bias_initializer=None,
                     name='H')

#onehot_labels = tf.one_hot(indices=tf.cast(label, tf.int32), depth=1)
onehot_labels = tf.reshape(label, [1, 1])
#loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=y)

loss = tf.square(tf.reduce_sum(y - onehot_labels))
tf.summary.scalar('loss', loss)

train_op = tf.contrib.layers.optimize_loss(
    loss=loss,
    global_step=tf.contrib.framework.get_global_step(),
    learning_rate=0.001,
    optimizer="SGD")



init_op = tf.global_variables_initializer()


'''          Building/Loading training data          '''
test_df_pickle = 'test1_with_scores_vec.pkl'
train_df_pickle = 'train1_with_scores_vec.pkl'

if os.path.isfile(test_df_pickle) and os.path.isfile(train_df_pickle):
    print('loading pickles')
    test_df = pd.read_pickle(test_df_pickle)
    train_df = pd.read_pickle(train_df_pickle)
else:
    print('processing & pickling CSVs')
    train_df = pd.read_csv('./train1.csv', encoding="utf-8")
    test_df = pd.read_csv('./test1.csv', encoding="utf-8")
    train_df = train_df[['question1', 'question2', 'is_duplicate']]
    test_df = test_df[['question1', 'question2', 'is_duplicate']]

    print('building word-vec dictionary')
    with open('wiki.en.vec') as f:
        vec_dictionary = {}
        content = f.readline()
        for i in tqdm(range(100000)):
            content = f.readline()
            content = content.strip()
            content = content.split(' ')
            word = content.pop(0)
            vec_dictionary.update({word: [float(i) for i in content]})
    print('word-vec dictionary BUILT')

    print('test_df addScores begin')
    test_df = addScores(test_df)
    print('train_df addScores begin')
    train_df = addScores(train_df)

    print('pickling')
    test_df.to_pickle(test_df_pickle)
    train_df.to_pickle(train_df_pickle)
    print('pickling DONE')



log_dir = 'log'


with tf.Session() as sess:
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(
        os.path.join(log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
    writer.add_graph(sess.graph)
    sess.run(init_op)

    a_feed = np.random.rand(3, 300)
    b_feed = np.random.rand(5, 300)
    label_feed = np.array([0])

    start_time = time.time()

    for i in tqdm(range(len(train_df))):
        a_feed = test_df['question1_vecs'][i]
        b_feed = test_df['question2_vecs'][i]

        if len(a_feed) < 1 or len(b_feed) < 1:
            continue
        label_feed = np.array([test_df['is_duplicate'][i]])
        summ, train_op1, losss, y2, one_hots = \
            sess.run([summaries, train_op, loss, y, onehot_labels], {a: a_feed, b: b_feed, label: label_feed})
        if i % 200 == 0:
            writer.add_summary(summ, global_step=i)
        if i % 1000 == 0:
            print('it: ', i, losss)
            #start_time = time.time()
