"""
- This file preprocesses the Bank Marketing dataset.
- You can access this dataset from the website
    'https://archive.ics.uci.edu/ml/datasets/bank+marketing'
- We choose the dataset 'bank-full.csv'
"""

import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# start preprocess
print('Start preprocess dataset bank...')

# Set the seed to ensure reproducibility
np.random.seed(821)
tf.random.set_seed(821)

# load dataset
data_path = '../datasets/raw/bank-full.csv'
df = pd.read_csv(data_path, sep=';')

# replace the unknown value with column's mode
df[df == 'unknown'] = np.nan
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# encode categorical attributes to integers
data = df.values
cat_index_list = []
for col in df.columns:
    if df[col].dtypes == object:
        cat_index_list.append(df.columns.get_loc(col))
for i in cat_index_list:
    category = np.unique(data[:, i])
    indices = tf.range(len(category), dtype=tf.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(category, indices)
    table = tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=1)
    data[:, i] = keras.layers.Lambda(lambda cats: table.lookup(cats))(data[:, i])

# encode numerical attributes with binning method
num_index_list = [0, 5, 9, 10, 11, 12, 13, 14]
age_bins = [15, 25, 45, 65, 120]
balance_bins = [-1e4] + [np.percentile(data[:, 5], percent, axis=0) for percent in [25, 50, 75]] + [2e5]
day_bins = [0, 10, 20, 31]
month_bins = [-1, 2, 5, 8, 11]
duration_bins = [-1.0] + [np.percentile(data[:, 11], percent, axis=0) for percent in [25, 50, 75]] + [6e3]
campaign_bins = [0.0] + [np.percentile(data[:, 12], percent, axis=0) for percent in [25, 50, 75]] + [1e2]
pdays_bins = [-10.0] + [np.percentile(data[:, 13], percent, axis=0) for percent in [25, 50, 75]] + [1e3]
previous_bins = [-1.0] + [np.percentile(data[:, 14], percent, axis=0) for percent in [25, 50, 75]] + [3e2]
bins_list = [age_bins, balance_bins, day_bins, month_bins, duration_bins, campaign_bins, pdays_bins, previous_bins]
for index, bins in zip(num_index_list, bins_list):
    data[:, index] = np.digitize(data[:, index], bins, right=True)

# save the dataset
file_path = '../datasets/bank'
alphabet = list(string.ascii_lowercase)[:data.shape[1]]
bank_data = pd.DataFrame(data, columns=alphabet)
bank_data.to_csv(file_path, index=False)


# finish preprocess
print('Dataset bank preprocess finished...')






