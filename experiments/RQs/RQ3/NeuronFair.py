"""
Algorithm NeuronFair.
- RQ3
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /baseline/NeuronFair.py.
"""

import sys
import time
import joblib


sys.path.append('..')
from utils import utils


# complete IDI generation of NeuronFair
def find_biased_neurons(dataset_name, model):
    start_time = time.time()

    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']
    layer_index, neurons = utils.find_biased_layer_and_neurons(x, model)

    end_time = time.time()
    execute_time = end_time - start_time

    return neurons, execute_time, layer_index




