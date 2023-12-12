"""
Algorithm DICE.
- RQ1
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /baseline/DICE.py.
"""

import sys
import time
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append('..')
from utils import utils
from utils import DICE_utils
from experiments.logfile.InfoLogger import InfoLogger


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
    all_id = np.empty(shape=(0, num_attrs))
    g_num = len(seeds)
    l_number = 0
    finish = False

    for i in range(g_num):
        if time.time() - start_time > timeout:
            break
        if finish:
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
                if len(g_id) >= 1000:
                    finish = True
                    break
            predict = DICE_utils.generate_predict(similar_x1_set, model)
            cluster = DICE_utils.clustering(predict, similar_x1_set, epsilon)
            x1, x2 = DICE_utils.global_sample_select(cluster)

            if is_idi and (len(cluster) - 1 >= 2):
                l_id, l_num = local_generation(x1, num_attrs, protected_attrs, constraint, model, start_time, timeout,
                                               l_num=10, s_l=1.0, epsilon=1e-6)
                l_number += l_num
                if l_id.size > 0:
                    all_id = np.vstack((all_id, l_id))

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

    all_id = np.array(list(set([tuple(i) for i in all_id])))
    l_nd_num = len(all_id)
    g_num = len(g_id)
    g_id = np.array(list(set([tuple(i) for i in g_id])))
    g_nd_num = len(g_id)
    if g_id.size > 0:
        all_id = np.vstack((all_id, g_id))
    all_id = np.array(list(set([tuple(i) for i in all_id])))
    return all_id, g_nd_num, g_num, l_nd_num, l_number


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
    l_num = len(l_id)
    l_id = np.array(list(set([tuple(i) for i in l_id])))
    return l_id, l_num


# fairness testing
def fairness_testing(clusters, num_attrs, protected_attrs, constraint, model, len_x, timeout,
                     max_global, max_iter, s_g, epsilon):
    inputs = DICE_utils.seed_test_input(clusters, min(max_global, len_x))
    start_time = time.time()
    all_id, g_nd_num, g_num, l_nd_num, l_num = global_generation(inputs, num_attrs, protected_attrs, constraint, model,
                                                                 start_time, timeout, max_iter, s_g, epsilon)

    return all_id, g_nd_num, g_num, l_nd_num, l_num


# complete IDI generation of DICE
def individual_discrimination_generation(dataset_name, config, model, c_num=4, timeout=6000):
    # logger Info
    start_time = time.time()
    logger = InfoLogger()

    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']
    labels = cluster_data['cluster_labels']

    num_attrs = len(x[0])
    len_x = len(x)
    clusters = [[] for _ in range(c_num)]
    for i, label in enumerate(labels):
        clusters[label].append(x[i])

    all_id, g_nd_num, g_num, l_nd_num, l_num = fairness_testing(clusters, num_attrs, config.protected_attrs,
                                                                config.constraint, model, len_x, timeout,
                                                                max_global=1000, max_iter=10, s_g=1.0, epsilon=0.025)

    logger.set_all_non_duplicate_number(len(all_id))
    logger.set_all_number(g_num + l_num)
    logger.set_global_number(g_num)
    logger.set_global_non_duplicate_number(g_nd_num)
    logger.set_local_number(l_num)
    logger.set_local_non_duplicate_number(l_nd_num)
    logger.set_total_time(time.time()-start_time)

    return logger
