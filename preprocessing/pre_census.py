"""
- This file preprocesses the Census Income dataset.
- You can access this dataset from the website
    'https://www.kaggle.com/vivamoto/us-adult-income-update?select=census.csv'
- We choose the dataset 'census.csv'
"""

import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# start preprocess
print('Start preprocess dataset census...')

# Set the seed to ensure reproducibility
np.random.seed(821)
tf.random.set_seed(821)

# load dataset
data_path = '../datasets/raw/census.csv'
df = pd.read_csv(data_path)

# drop the unnecessary value
df = df.drop(['education'], axis=1)

# replace the unknown value with column's mode
df[df == '?'] = np.nan
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
num_index_list = [0, 2, 9, 10, 11]
age_bins = [15, 25, 45, 65, 120]
fnlwgt_bins = [1e4] + [np.percentile(data[:, 2], percent, axis=0) for percent in (np.arange(0.0, 99, 1))] + [2e6]
capital_gain_bins = [-1, 0, 99998, 100000]
capital_loss_bins = [-1, 0, 99998, 100000]
hours_per_week_bins = [0, 25, 40, 60, 168]
bins_list = [age_bins, fnlwgt_bins, capital_gain_bins, capital_loss_bins, hours_per_week_bins]
for index, bins in zip(num_index_list, bins_list):
    data[:, index] = np.digitize(data[:, index], bins, right=True)

# save the dataset
file_path = '../datasets/census'
alphabet = list(string.ascii_lowercase)[:data.shape[1]]
census_data = pd.DataFrame(data, columns=alphabet)
census_data.to_csv(file_path, index=False)

# finish preprocess
print('Dataset census preprocess finished...')


