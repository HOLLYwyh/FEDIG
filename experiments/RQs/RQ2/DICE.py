"""
Algorithm DICE.
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /baseline/DICE.py.
"""

import sys
import time
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append('../..')
from utils import utils
from utils import DICE_utils


# compute_grad()
def compute_grad(x, model, y=None, loss_func=keras.losses.binary_crossentropy):
    x = tf.constant([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_predict = model(x)
        if y is None:
            y = tf.cast(y_predict > 0.5, dtype=tf.float32)
        else:
            y = tf.constant([[y]], dtype=tf.float32)
        loss = loss_func(y_predict, y)
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()


# global_generation of DICE
def global_generation(seeds, num_attrs, protected_attrs, constraint, model, start_time,
                      timeout, max_iter, s_g, epsilon):
    g_id = np.empty(shape=(0, num_attrs))
    g_num = len(seeds)

    for i in range(g_num):
        if time.time() - start_time > timeout:
            break
        x1 = seeds[i].copy()
        for _ in range(max_iter + 1):
            if time.time() - start_time > timeout:
                break
            is_idi = False
            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x1, similar_x1_set, model):
                g_id = np.append(g_id, [x1], axis=0)
                is_idi = True
            predict = DICE_utils.generate_predict(similar_x1_set, model)
            cluster = DICE_utils.clustering(predict, similar_x1_set, epsilon)
            x1, x2 = DICE_utils.global_sample_select(cluster)

            if is_idi and (len(cluster) - 1 >= 2):
                l_id = local_generation(x1, num_attrs, protected_attrs, constraint, model,
                                        start_time, timeout, l_num=10, s_l=1.0, epsilon=1e-6)
                g_id = np.append(g_id, l_id, axis=0)

            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction = np.zeros_like(x1)
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attr in range(num_attrs):
                if attr not in protected_attrs and sign_grad1[attr] == sign_grad2[attr]:
                    direction[attr] = sign_grad1[attr]
            x1 = x1 + direction * s_g
            x1 = utils.clip(x1, constraint)

    g_id = np.array(list(set([tuple(i) for i in g_id])))
    return g_id


# local_generation of DICE
def local_generation(x1, num_attrs, protected_attrs, constraint, model,
                     start_time, timeout, l_num, s_l, epsilon):
    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attrs))
    for _ in range(l_num):
        if time.time() - start_time > timeout:
            break
        x1_copy = x1.copy()
        similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
        x2 = utils.find_idi_pair(x1, similar_x1_set, model)
        grad1 = compute_grad(x1, model)
        grad2 = compute_grad(x2, model)
        p = utils.normalization(grad1, grad2, protected_attrs, epsilon)
        a = utils.random_pick(p)
        d = direction[utils.random_pick([0.5, 0.5])]
        x1[a] = x1[a] + d * s_l
        x1 = utils.clip(x1, constraint)
        similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
        if utils.is_discriminatory(x1, similar_x1_set, model):
            l_id = np.append(l_id, [x1], axis=0)
        else:
            x1 = x1_copy
    l_id = np.array(list(set([tuple(i) for i in l_id])))
    return l_id


# fairness testing
def fairness_testing(clusters, num_attrs, protected_attrs, constraint, model, len_x, timeout,
                     max_global, max_iter, s_g, epsilon):
    all_id = np.empty(shape=(0, num_attrs))
    inputs = DICE_utils.seed_test_input(clusters, min(max_global, len_x))
    start_time = time.time()
    g_id = global_generation(inputs, num_attrs, protected_attrs, constraint, model, start_time,
                             timeout, max_iter, s_g, epsilon)
    all_id = np.append(all_id, g_id, axis=0)

    all_id = np.array(list(set([tuple(i) for i in all_id])))
    return all_id


# complete IDI generation of DICE
def individual_discrimination_generation(dataset_name, config, model, c_num=4, timeout=600):
    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']
    labels = cluster_data['cluster_labels']

    num_attrs = len(x[0])
    len_x = len(x)
    clusters = [[] for _ in range(c_num)]
    for i, label in enumerate(labels):
        clusters[label].append(x[i])

    all_id = fairness_testing(clusters, num_attrs, config.protected_attrs, config.constraint, model, len_x,
                              timeout, max_global=1000, max_iter=10, s_g=1.0, epsilon=0.025)
    return all_id
