"""
 This paper will determine the parameter η through experiments.
 - The value of η can be: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
 - 0.7, 0.8, 0.9, 1.0
"""

import sys
import time
import pandas as pd
from keras.models import load_model

sys.path.append('..')
from FEDIG import FEDIG
from utils import config

# use credit dataset as example
path = '../models/trained_models/credit_model.h5'
data_path = '../datasets/credit'

df = pd.read_csv(data_path)
data = df.values[:, 1:]

model = load_model(path)

# save the logging data
file_path = 'logfile/parameter/eta.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['number', 'time'])

# experiment begins
for i in range(0, 11):
    start_time = time.time()
    decay = i * 0.1
    all_id = FEDIG.individual_discrimination_generation('credit', config.Credit, model, decay=decay)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"=======================Round{i+1}=======================")
    print(len(all_id))
    print(execution_time)

    new_data = pd.DataFrame({'number': [len(all_id)], 'time': [execution_time]}, index=[0])
    df = df.append(new_data, ignore_index=True)

df = df.append(pd.DataFrame({'number': ['-'], 'time': ['-']}, index=[0]), ignore_index=True)
df.to_csv(file_path, index=False)
# experiment ends
