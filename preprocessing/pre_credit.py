"""
- This file preprocesses the German Credit dataset.
- You can access this dataset from the website
    'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q8MAW8'
- We choose the dataset 'proc_german_num_02 withheader-2.csv'
"""

import string
import numpy as np
import pandas as pd
import tensorflow as tf

# start preprocess
print('Start preprocess dataset credit...')

# Set the seed to ensure reproducibility
np.random.seed(821)
tf.random.set_seed(821)

# load dataset
data_path = '../datasets/raw/proc_german_num_02 withheader-2.csv'
df = pd.read_csv(data_path)

# preprocess data
data = df.values
data[:, 0] = data[:, 0] == 1
loan_nurnmonth_bins = [0] + [np.percentile(data[:,  2], percent, axis=0) for percent in [25, 50, 75]] + [80]
creditamt_bins = [0] + [np.percentile(data[:, 4], percent, axis=0) for percent in [25, 50, 75]] + [200]
age_in_years_bins = [15, 25, 45, 65, 120]
indexes_list = [2, 4, 10]
bins_list = [loan_nurnmonth_bins, creditamt_bins, age_in_years_bins]
for index, bins in zip(indexes_list, bins_list):
    data[:, index] = np.digitize(data[:, index], bins, right=True)

# save the dataset
file_path = '../datasets/credit'
alphabet = list(string.ascii_lowercase)[:data.shape[1]]
credit_data = pd.DataFrame(data, columns=alphabet)
credit_data.to_csv(file_path, index=False)


# finish preprocess
print('Dataset credit preprocess finished...')

