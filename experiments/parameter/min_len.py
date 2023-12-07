"""
This paper will determine the parameter min_len through experiments.
- The value of min_len can be: 250, 500, 750, 1000, 1250, 1500, 1750, 2000
"""

import sys
import time
import pandas as pd
from tensorflow.keras.models import load_model

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
file_path = 'logfile/parameter/min_len.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['number', 'time'])

# experiment begins
len_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
for min_len in len_list:
    start_time = time.time()
    all_id = FEDIG.individual_discrimination_generation('credit', config.Credit, model, decay=0.2,
                                                        c_num=4, min_len=min_len, delta1=0.1, delta2=0.2)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"================={min_len}=================")
    print(len(all_id))
    print(execution_time)

    new_data = pd.DataFrame({'number': [len(all_id)], 'time': [execution_time]}, index=[0])
    df = df.append(new_data, ignore_index=True)

df = df.append(pd.DataFrame({'number': ['-'], 'time': ['-']}, index=[0]), ignore_index=True)
df.to_csv(file_path, index=False)
# experiment ends


