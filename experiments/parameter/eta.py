"""
 This paper will determine the parameter η through experiments.
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

# experiment ends
