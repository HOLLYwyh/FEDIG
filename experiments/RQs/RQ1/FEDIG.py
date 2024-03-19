"""
Algorithm FEDIG.
- RQ1
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /FEDIG/FEDIG.py.
"""

import sys
import time
import joblib
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
    finish = False

    for i in range(g_num):
        if finish:
            break

        x0 = seeds[i].copy()
        potential_x_list = [x0]
        grad1 = np.zeros_like(x0).astype(float)
        grad2 = np.zeros_like(x0).astype(float)
        for _ in range(max_iter):
            if finish:
                break

            is_discriminatory = False
            for x1 in potential_x_list:
                similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
                if utils.is_discriminatory(x1, similar_x1_set, model):
                    is_discriminatory = True
                    g_id = np.append(g_id, [x1], axis=0)
                    if len(g_id) >= 1000:
                        finish = True
                        break

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
    g_num = len(g_id)
    g_id = np.array(list(set([tuple(i) for i in g_id])))
    return g_id, len(g_id), g_num


# local generation of FEDIG
def local_generation(num_attrs, g_id, protected_attrs, constraint, model, irrelevant_features,
                     decay, s_l):
    l_id = np.empty(shape=(0, num_attrs))

    for x1 in g_id:
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

    l_num = len(l_id)
    l_id = np.array(list(set([tuple(i) for i in l_id])))

    return l_id, len(l_id), l_num


# complete IDI generation of FEDIG
def individual_discrimination_generation(dataset_name, config, model, decay=0.5, min_len=1000):
    # logger Info
    start_time = time.time()
    logger = InfoLogger()

    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']

    num_attrs = len(x[0])
    all_biased_features = FEDIG_utils.sort_biased_features(x, num_attrs, model,
                                                           config.protected_attrs, config.constraint, min_len)
    irrelevant_features, optimal_features = FEDIG_utils.spilt_biased_features(all_biased_features)

    all_id = np.empty(shape=(0, num_attrs))
    np.random.shuffle(x)

    g_id, g_nd_num, g_num = global_generation(x, num_attrs, config.protected_attrs, config.constraint, model,
                                              optimal_features, decay, max_iter=10, s_g=1.0)

    l_id, l_nd_num, l_num = local_generation(num_attrs, g_id, config.protected_attrs, config.constraint, model,
                                             irrelevant_features, decay, s_l=1.0)

    part_id = np.append(g_id, l_id, axis=0)
    all_id = np.append(all_id, part_id, axis=0)

    all_id = np.array(list(set([tuple(i) for i in all_id])))

    logger.set_all_non_duplicate_number(len(all_id))
    logger.set_all_number(g_num + l_num)
    logger.set_global_number(g_num)
    logger.set_global_non_duplicate_number(g_nd_num)
    logger.set_local_number(l_num)
    logger.set_local_non_duplicate_number(l_nd_num)
    logger.set_total_time(time.time()-start_time)
    return logger




