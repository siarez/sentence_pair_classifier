import numpy as np
import tensorflow as tf
import os.path
import pandas as pd
import nltk
from tqdm import tqdm
from pair_classifier_model import Model


def add_embedded_sentences(df, word_vec_dict):
    """Here sentences are sanitized, tokenized, and vector representations of question are added to the dataframe."""
    q1_vec = []
    q2_vec = []
    for q1q2 in tqdm(zip(df['question1'], df['question2'])):
        if type(q1q2[0]) is not str:
            q1_clean = str(q1q2[0])
        else:
            q1_clean = nltk.word_tokenize(q1q2[0].encode('ascii', 'ignore').decode('utf-8', 'ignore').
                                          lower().replace('[^0-9a-zA-Z ]+', ''))

        sentence1_vecs = [word_vec_dict[w] for w in q1_clean if w in word_vec_dict]
        # This makes sure a sentence is at least represented by a zeros vector,
        # if none of the words where found in the dictionary.
        q1_vec.append(sentence1_vecs) if (len(sentence1_vecs) >= 1) \
            else q1_vec.append(np.zeros((1, 300), dtype=float))

        if type(q1q2[1]) is not str:
            q2_clean = str(q1q2[1])
        else:
            q2_clean = nltk.word_tokenize(q1q2[1].encode('ascii', 'ignore').decode('utf-8', 'ignore').
                                          lower().replace('[^0-9a-zA-Z ]+', ''))
        sentence2_vecs = [word_vec_dict[w] for w in q2_clean if w in word_vec_dict]
        q2_vec.append(sentence2_vecs) if (len(sentence2_vecs) >= 1) \
            else q2_vec.append(np.zeros((1, 300), dtype=float))

    df['question1_vecs'] = pd.Series(q1_vec)
    df['question2_vecs'] = pd.Series(q2_vec)
    return df

checkpoint_dir = 'save'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
assert ckpt, 'No checkpoint found'
assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
print('Using checkpoint: {}'.format(ckpt.model_checkpoint_path[-2:]))

real_test_filename = 'test.csv'
real_test_df_pickle = 'test_preprocessed.pkl'
word_embeddings = 'glove.840B.300d.txt'

if os.path.isfile(real_test_df_pickle):
    print('Loading pickled dataframe: %s' % real_test_df_pickle)
    real_test_df = pd.read_pickle(real_test_df_pickle)
else:
    print('processing & pickling %s' % real_test_filename)
    real_test_df = pd.read_csv(real_test_filename, encoding='utf-8')
    real_test_df = real_test_df[['test_id', 'question1', 'question2']]

    print('building word-vec dictionary')
    with open(word_embeddings) as f:
        vec_dictionary = {}
        content = f.readline()
        for i in tqdm(range(100000)):
            content = f.readline()
            content = content.strip()
            content = content.split(' ')
            word = content.pop(0)
            vec_dictionary.update({word: [float(i) for i in content]})

    print('add_embedded_sentences')
    test_df = add_embedded_sentences(real_test_df, vec_dictionary)

    print('pickling')
    real_test_df.drop(['question1', 'question2'], axis=1, inplace=True)  # to save space and time
    real_test_df.to_pickle(real_test_df_pickle)
    print('pickling DONE')


model = Model('INFER')
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, ckpt.model_checkpoint_path)
    results = []
    results_prob = []
    # Start inference on the real test data
    for i in tqdm(range(len(real_test_df))):
        a_feed = real_test_df['question1_vecs'][i]
        b_feed = real_test_df['question2_vecs'][i]
        is_duplicate, prob = sess.run([model.classes, model.probabilities], {model.a: a_feed, model.b: b_feed})
        results.append(is_duplicate[0][0])
        results_prob.append(prob[0][1])
    del real_test_df

submission = pd.DataFrame({'test_id': range(len(results)), 'is_duplicate': results})
submission.to_csv('submission_{}.csv'.format(ckpt.model_checkpoint_path[-2:]), encoding='utf-8',
                  columns=['test_id', 'is_duplicate'], index=False)

submission_prob = pd.DataFrame({'test_id': range(len(results_prob)), 'is_duplicate': results_prob})
submission_prob.to_csv('submission_prob_{}.csv'.format(ckpt.model_checkpoint_path[-2:]), encoding='utf-8',
                       columns=['test_id', 'is_duplicate'], index=False)
