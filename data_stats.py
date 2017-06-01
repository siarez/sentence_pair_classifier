import os.path
import pandas as pd
from tqdm import tqdm


def calc_stats(df):

    q1_vec_len = []
    q2_vec_len = []
    for q1q2 in tqdm(zip(df['question1_vecs'], df['question2_vecs'])):
        q1_vec_len.append(len(q1q2[0]))
        q2_vec_len.append(len(q1q2[1]))

    q1_vec_len = pd.Series(q1_vec_len)
    q2_vec_len = pd.Series(q2_vec_len)

    q1_vec_len_mean = q1_vec_len.mean()
    q2_vec_len_mean = q2_vec_len.mean()
    q1_vec_len_std = q1_vec_len.std()
    q2_vec_len_std = q2_vec_len.std()
    q1_vec_len_max = q1_vec_len.max()
    q2_vec_len_max = q2_vec_len.max()
    print('q1_vec_len_mean: ', q1_vec_len_mean)
    print('q2_vec_len_mean: ', q2_vec_len_mean)
    print('q1_vec_len_std: ', q1_vec_len_std)
    print('q2_vec_len_std: ', q2_vec_len_std)
    print('q1_vec_len_max: ', q1_vec_len_max)
    print('q2_vec_len_max: ', q2_vec_len_max)

    return


pickled_dataframes = [ 'test1_with_scores_vec.pkl', 'train1_with_scores_vec.pkl', 'test_with_scores_vec.pkl']

for file_name in pickled_dataframes:
    if os.path.isfile(file_name):
        print('loading %s: ' % file_name)
        dataframe = pd.read_pickle(file_name)
        print('calculating stats')
        calc_stats(dataframe)