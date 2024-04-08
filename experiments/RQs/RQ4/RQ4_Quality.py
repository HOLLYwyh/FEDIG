"""
- RQ4
- Quality of FEDIG for images
"""

import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import FEDIG_img_Quality

sys.path.append('..')
from keras.models import load_model

# log
logging.basicConfig(filename='Quality_01-100.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load models
ca_model_path = '../models/trained_models/celeba_model.h5'
ff_model_path = '../models/trained_models/FairFace_model.h5'
ca_model = load_model(ca_model_path)
ff_model = load_model(ff_model_path)

# load instances
instance_num = 1000
ca_image_path = '../datasets/celebA/img_align_celeba/'
ff_image_path = '../datasets/FairFace/val/'
ca_file_names = np.array(random.sample(os.listdir(ca_image_path), instance_num))
ff_file_names = np.array(
    random.sample(sorted(os.listdir(ff_image_path), key=lambda x: int(x.split('.')[0])), instance_num))

# save the logging data
file_path = 'logfile/RQs/RQ4_Quality-01-100.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['data', 'all_num', 'all_nd_num', 'g_num', 'g_nd_num', 'l_num', 'l_nd_num', 'time'])
    df.to_csv(file_path, index=False)

# idi generation
# ca
print("===============================")
print("ca begins....")
print("===============================")
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['ca'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
ca_logger = FEDIG_img_Quality.cnn_idi_generation(ca_file_names, ca_model, ca_image_path, (256, 256))
df = df.append(
    pd.DataFrame({'data': ['ca'], 'all_num': [ca_logger.all_number], 'all_nd_num': [ca_logger.all_non_duplicate_number],
                  'g_num': [ca_logger.global_number], 'g_nd_num': [ca_logger.global_non_duplicate_number],
                  'l_num': [ca_logger.local_number], 'l_nd_num': [ca_logger.local_non_duplicate_number],
                  'time': [ca_logger.total_time]},
                 index=[0]), ignore_index=True)
print(f"==========={ca_logger.all_non_duplicate_number}==========={ca_logger.all_number}"
      f"==========={ca_logger.total_time}===========")
df.to_csv(file_path, index=False)

# ff
print("===============================")
print("ff begins....")
print("===============================")
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['ff'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
ff_logger = FEDIG_img_Quality.cnn_idi_generation(ff_file_names, ff_model, ff_image_path, (224, 224))
df = df.append(
    pd.DataFrame({'data': ['ff'], 'all_num': [ff_logger.all_number], 'all_nd_num': [ff_logger.all_non_duplicate_number],
                  'g_num': [ff_logger.global_number], 'g_nd_num': [ff_logger.global_non_duplicate_number],
                  'l_num': [ff_logger.local_number], 'l_nd_num': [ff_logger.local_non_duplicate_number],
                  'time': [ff_logger.total_time]},
                 index=[0]), ignore_index=True)
print(f"==========={ff_logger.all_non_duplicate_number}==========={ff_logger.all_number}"
      f"==========={ff_logger.total_time}===========")

df.to_csv(file_path, index=False)
print("experiment ends......")
# experiment ends
