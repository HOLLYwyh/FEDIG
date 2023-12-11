"""
Algorithm FEDIG.
- RQ4
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /FEDIG/FEDIG.py.
"""

import sys
import time
import joblib
import random
import numpy as np
import tensorflow as tf
sys.path.append('..')
from utils import utils
from utils import FEDIG_utils
from experiments.logfile.InfoLogger import InfoLogger


# compute the gradient of model predictions w.r.t. input attributes
def compute_grad(x, model):
    x = tf.constant([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_predict = model(x)
    gradient = tape.gradient(y_predict, x)
    return gradient[0].numpy()


# global generation of FEDIG
def global_generation(seeds, num_attrs, protected_attrs, constraint, model, optimal_features,
                      decay, max_iter, s_g):
    g_id = np.empty(shape=(0, num_attrs))
    g_num = len(seeds)
    print('Global generation starts...')
    print(f'Total batch number: {g_num}')

    for i in range(g_num):
        print(f'batch number{i+1}/{g_num}')
        x0 = seeds[i].copy()
        potential_x_list = [x0]
        grad1 = np.zeros_like(x0).astype(float)
        grad2 = np.zeros_like(x0).astype(float)
        for _ in range(max_iter):
            is_discriminatory = False
            for x1 in potential_x_list:
                similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
                if utils.is_discriminatory(x1, similar_x1_set, model):
                    g_id = np.append(g_id, [x1], axis=0)
                    is_discriminatory = True
            if is_discriminatory:
                break

            potential_x1_list = []
            for x1 in potential_x_list:
                similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
                x2 = FEDIG_utils.find_boundary_pair(x1, similar_x1_set, model)
                grad1 = decay * grad1 + compute_grad(x1, model)
                grad2 = decay * grad2 + compute_grad(x2, model)
                direction = np.sign(grad1 + grad2)
                for attr in protected_attrs:
                    direction[attr] = 0
                potential_x1_list.extend(FEDIG_utils.get_potential_global_x(x1, direction, optimal_features, constraint, s_g))
            potential_x_list = np.array(list(set([tuple(i) for i in potential_x1_list])))

    g_id = np.array(list(set([tuple(i) for i in g_id])))
    print('Global generation finishes...')
    return g_id


# local generation of FEDIG
def local_generation(num_attrs, g_id, protected_attrs, constraint, model, irrelevant_features,
                     decay, s_l):
    l_id = np.empty(shape=(0, num_attrs))
    batch_num = 0
    total_num = len(g_id)
    print('Local generation starts...')
    print(f'Total batch number:{total_num}')

    for x1 in g_id:
        batch_num += 1
        print(f'batch number{batch_num + 1}/{total_num}')
        grad1 = np.zeros_like(x1).astype(float)
        grad2 = np.zeros_like(x1).astype(float)

        similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
        x2 = utils.find_idi_pair(x1, similar_x1_set, model)
        grad1 = decay * grad1 + compute_grad(x1, model)
        grad2 = decay * grad2 + compute_grad(x2, model)
        p = FEDIG_utils.normalization(grad1, grad2, protected_attrs, irrelevant_features, 1e-6)
        a = utils.random_pick(p)
        grad_sign = np.sign(grad1 + grad2)

        potential_x1_list = FEDIG_utils.get_potential_local_x(x1, grad_sign, a, irrelevant_features, constraint, s_l)

        for x in potential_x1_list:
            similar_x_set = utils.get_similar_set(x, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x, similar_x_set, model):
                l_id = np.append(l_id, [x], axis=0)

    l_id = np.array(list(set([tuple(i) for i in l_id])))
    print('Local generation finishes...')
    return l_id


# complete IDI generation of FEDIG
def individual_discrimination_generation(dataset_name, config, model, decay=0.5,
                                         c_num=4, min_len=1000, delta1=0.10, delta2=0.20):

    # logger Info
    logger = InfoLogger()
    start_time = time.time()

    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']
    original_labels = cluster_data['cluster_labels']

    # We use 1,000 instances when the number of dataset exceeds 1000.
    if len(original_labels) > 1500:
        labels = random.sample(list(original_labels), 1500)
    else:
        labels = original_labels

    num_attrs = len(x[0])
    all_biased_features = FEDIG_utils.sort_biased_features(x, num_attrs, model,
                                                           config.protected_attrs, config.constraint, min_len)
    irrelevant_features, optimal_features = FEDIG_utils.spilt_biased_features(all_biased_features, delta1, delta2)

    all_id = np.empty(shape=(0, num_attrs))
    clusters = [[] for _ in range(c_num)]
    for i, label in enumerate(labels):
        clusters[label].append(x[i])

    for i in range(len(clusters)):
        g_id = global_generation(clusters[i], num_attrs, config.protected_attrs, config.constraint, model,
                                 optimal_features, decay, max_iter=10, s_g=1.0)

        l_id = local_generation(num_attrs, g_id, config.protected_attrs, config.constraint, model,
                                irrelevant_features, decay, s_l=1.0)

        part_id = np.append(g_id, l_id, axis=0)
        all_id = np.append(all_id, part_id, axis=0)

    all_id = np.array(list(set([tuple(i) for i in all_id])))

    end_time = time.time()
    logger.set_total_time(end_time - start_time)

    return all_id, logger





