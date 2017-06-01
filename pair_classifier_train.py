import numpy as np
import tensorflow as tf
import nltk
import time
import os.path
import pandas as pd
from tqdm import tqdm
from pair_classifier_model import Model


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def add_embedded_sentences(df, word_vec_dict):
    # '[^0-9a-zA-Z ?.]+'
    q1_vec = []
    q2_vec = []
    for q1q2 in tqdm(zip(df['question1'], df['question2'])):
        if type(q1q2[0]) is not str:
            q1_clean = str(q1q2[0])
        else:
            q1_clean = nltk.word_tokenize(q1q2[0].encode('ascii', 'ignore').decode('utf-8', 'ignore').lower().replace('[^0-9a-zA-Z ]+', ''))

        sentence1_vecs = [word_vec_dict[w] for w in q1_clean if w in word_vec_dict]
        # This makes sure a sentence is atleast represented by a zeros vector,
        # if none of the words where found in the dictionary.
        q1_vec.append(sentence1_vecs) if (len(sentence1_vecs) >= 1) \
            else q1_vec.append(np.zeros((1, 300), dtype=float))

        if type(q1q2[1]) is not str:
            q2_clean = str(q1q2[1])
        else:
            q2_clean = nltk.word_tokenize(q1q2[1].encode('ascii', 'ignore').decode('utf-8', 'ignore').lower().replace('[^0-9a-zA-Z ]+', ''))
        sentence2_vecs = [word_vec_dict[w] for w in q2_clean if w in word_vec_dict]
        q2_vec.append(sentence2_vecs) if (len(sentence2_vecs) >= 1) \
            else q2_vec.append(np.zeros((1, 300), dtype=float))

    df['question1_vecs'] = pd.Series(q1_vec)
    df['question2_vecs'] = pd.Series(q2_vec)
    return df


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
    with open('glove.840B.300d.txt') as f:
        vec_dictionary = {}
        content = f.readline()
        for i in tqdm(range(100000)):
            content = f.readline()
            content = content.strip()
            content = content.split(' ')
            word = content.pop(0)
            vec_dictionary.update({word: [float(i) for i in content]})
    print('word-vec dictionary BUILT')

    print('test_df add_embedded_sentences')
    test_df = add_embedded_sentences(test_df, vec_dictionary)
    print('train_df add_embedded_sentences')
    train_df = add_embedded_sentences(train_df, vec_dictionary)

    print('pickling')
    test_df.to_pickle(test_df_pickle)
    train_df.to_pickle(train_df_pickle)
    print('pickling DONE')


log_dir = 'log'
save_dir = 'save'
model = Model('TRAIN')
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    #summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
    writer_eval = tf.summary.FileWriter(os.path.join(log_dir, 'eval', time.strftime("%Y-%m-%d-%H-%M-%S")))
    writer.add_graph(sess.graph)
    sess.run(init_op)
    # Create a saver object which will save all the variables
    saver = tf.train.Saver()

    a_feed = np.random.rand(3, 300)
    b_feed = np.random.rand(5, 300)
    label_feed = np.array([0])

    for epoch in range(10):
        for i in tqdm(range(len(train_df))):
        #for i in tqdm(range(5000)):
            a_feed = train_df['question1_vecs'][i]
            b_feed = train_df['question2_vecs'][i]

            if len(a_feed) < 1 or len(b_feed) < 1:
                continue
            label_feed = np.array([train_df['is_duplicate'][i]])
            #label_feed = np.random.randint(2, size=1)
            #a_feed[0][0] = float(label_feed[0]) #cheat hint

            summ, train_op, loss = \
                sess.run([model.loss_summary, model.train_op, model.loss], {model.a: a_feed, model.b: b_feed, model.label: label_feed})
            if i % 200 == 0:
                writer.add_summary(summ, global_step=i * (1 + epoch))
            if i % 500 == 0:
                print('it: ', i, loss, )

            if i % 50000 == 0:
                print('\nRunning validation')
                sess.run(tf.assign(model.accuracy, tf.constant(0.0)))
                sess.run(tf.assign(model.validation_iter, tf.constant(1)))
                #test_results = []
                random_sample = test_df.sample(n=1000, replace=True)
                for j in (range(len(random_sample))):
                    a_feed = random_sample['question1_vecs'].values[j]
                    b_feed = random_sample['question2_vecs'].values[j]
                    label_feed = np.array([random_sample['is_duplicate'].values[j]])
                    is_duplicate_prediction, accuracy_summ, accuracy, my_iter = \
                        sess.run([model.classes, model.accuracy_summary_op, model.accuracy_op, model.validation_iter_op],
                                 {model.a: a_feed, model.b: b_feed, model.label: label_feed})
                writer_eval.add_summary(accuracy_summ, global_step=i * (1 + epoch))
                print(accuracy, my_iter)

        # save checkpoint after each epoch
        checkpoint_path = os.path.join(save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=epoch)
        print("model saved to {}".format(checkpoint_path))






