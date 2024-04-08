"""
This file is the experiments of RQ3:
    - Explanation of the unfairness with features(img).
"""

import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import FEDIG_img

sys.path.append('..')
from keras.models import load_model

# load models
ca_model_path = '../models/trained_models/celeba_model.h5'
ff_model_path = '../models/trained_models/FairFace_model.h5'
ca_model = load_model(ca_model_path)
ff_model = load_model(ff_model_path)

# load instances
instance_num = 10
random.seed(821)
ca_dsize = (256, 256)
ff_dsize = (224, 224)
ca_image_path = '../datasets/celebA/img_align_celeba/'
ff_image_path = '../datasets/FairFace/val/'
ca_file_names = np.array(random.sample(os.listdir(ca_image_path), instance_num))
ff_file_names = np.array(
    random.sample(sorted(os.listdir(ff_image_path), key=lambda x: int(x.split('.')[0])), instance_num))

# save the logging data
file_path = 'logfile/RQs/RQ3_img.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['explanation', 'total_time', 'time_list'])

# irrelevant features and biased features generation
# celebA
df = df.append(pd.DataFrame({'explanation': ['celebA'], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))
ca_ir_features, ca_bi_features, ca_total_time, ca_time_list = FEDIG_img.cnn_idi_generation(ca_file_names, ca_model,
                                                                                           ca_image_path, ca_dsize)
df = df.append(pd.DataFrame({'explanation': ['ca_bi_features'], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))
df = df.append(pd.DataFrame({'explanation': [ca_bi_features], 'total_time': [ca_total_time], 'time_list': [ca_time_list]}, index=[0]))
df = df.append(pd.DataFrame({'explanation': ['ca_ir_features'], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))
df = df.append(pd.DataFrame({'explanation': [ca_ir_features], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))

# FairFace
df = df.append(pd.DataFrame({'explanation': ['FairFace'], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))
ff_ir_features, ff_bi_features, ff_total_time, ff_time_list = FEDIG_img.cnn_idi_generation(ff_file_names, ff_model,
                                                                                           ff_image_path, ff_dsize)
df = df.append(pd.DataFrame({'explanation': ['ff_bi_features'], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))
df = df.append(pd.DataFrame({'explanation': [ff_bi_features], 'total_time': [ff_total_time], 'time_list': [ff_time_list]}, index=[0]))
df = df.append(pd.DataFrame({'explanation': ['ff_ir_features'], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))
df = df.append(pd.DataFrame({'explanation': [ff_ir_features], 'total_time': ['-'], 'time_list': ['-']}, index=[0]))

df.to_csv(file_path, index=False)
