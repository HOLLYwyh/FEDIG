"""
Algorithm FEDIG.
- RQ3.
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /FEDIG/FEDIG.py.
"""

import sys
import time
import joblib


sys.path.append('..')

from utils import FEDIG_utils


# find the biased feature.
def find_biased_features(dataset_name, config, model, min_len):
    start_time = time.time()
    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']

    num_attrs = len(x[0])
    all_biased_features = FEDIG_utils.sort_biased_features(x, num_attrs, model,
                                                           config.protected_attrs, config.constraint, min_len)
    end_time = time.time()
    execute_time = end_time - start_time

    return all_biased_features, execute_time
