import numpy as np
import tensorflow as tf
import os.path
import pandas as pd
from tqdm import tqdm



'''   Start inference on the real test data   '''
print('starting inference')
real_test_df_pickle = 'test_with_scores_vec.pkl'
real_test_df = pd.read_pickle(real_test_df_pickle)
results = []
for i in tqdm(range(len(real_test_df))):
    # for i in tqdm(range(60)):
    a_feed = real_test_df['question1_vecs'][i]
    b_feed = real_test_df['question2_vecs'][i]
    is_duplicate_prob = sess.run(y_prob, {a: a_feed, b: b_feed})
    results.append(is_duplicate_prob[0][1])
    if i % 1 == 0:
        print('it: ', i, is_duplicate_prob)
del real_test_df

submission = pd.DataFrame({'test_id': range(len(results)), 'is_duplicate': results})
submission.to_csv('submission.csv', encoding='utf-8', columns=['test_id', 'is_duplicate'], index=False)
