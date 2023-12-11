"""
Algorithm EIDIG.
- RQ1
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /baseline/EIDIG.py.
"""

import sys
import time
import joblib
import numpy as np
import tensorflow as tf
sys.path.append('..')
from utils import utils
from experiments.logfile.InfoLogger import InfoLogger


# compute the gradient of model predictions w.r.t input attributes
def compute_grad(x, model):
    x = tf.constant([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_predict = model(x)
    gradient = tape.gradient(y_predict, x)
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()


# global generation of EIDIG
def global_generation(seeds, num_attrs, protected_attrs, constraint, model, decay, max_iter, s_g):
    g_id = np.empty(shape=(0, num_attrs))
    g_num = len(seeds)
    finish = False

    for i in range(g_num):
        if finish:
            break

        x1 = seeds[i].copy()
        grad1 = np.zeros_like(x1).astype(float)
        grad2 = np.zeros_like(x1).astype(float)
        for _ in range(max_iter):
            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x1, similar_x1_set, model):
                g_id = np.append(g_id, [x1], axis=0)
                if len(g_id) >= 1000:
                    finish = True
                break
            x2 = utils.argmax(x1, similar_x1_set, model)
            grad1 = decay * grad1 + compute_grad(x1, model)
            grad2 = decay * grad2 + compute_grad(x2, model)
            direction = np.zeros_like(x1)
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attr in range(num_attrs):
                if attr not in protected_attrs and sign_grad1[attr] == sign_grad2[attr]:
                    direction[attr] = (-1) * sign_grad1[attr]
            x1 = x1 + s_g * direction
            x1 = utils.clip(x1, constraint)
    g_num = len(g_id)
    g_id = np.array(list(set([tuple(i) for i in g_id])))
    return g_id, len(g_id), g_num


# local generation of EIDIG
def local_generation(num_attrs, g_id, protected_attrs, constraint, model, update_interval, l_num, s_l, epsilon):
    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attrs))
    p0 = np.empty(shape=(0, num_attrs))
    p = np.empty(shape=(0, num_attrs))

    for x1 in g_id:
        x0 = x1.copy()
        counts = update_interval
        for _ in range(l_num):
            if counts == update_interval:
                similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
                x2 = utils.find_idi_pair(x1, similar_x1_set, model)
                grad1 = compute_grad(x1, model)
                grad2 = compute_grad(x2, model)
                p = utils.normalization(grad1, grad2, protected_attrs, epsilon)
                p0 = p.copy()
                counts = 0
            counts += 1
            a = utils.random_pick(p)
            x1[a] = x1[a] + direction[utils.random_pick([0.5, 0.5])] * s_l
            x1 = utils.clip(x1, constraint)
            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x1, similar_x1_set, model):
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x0.copy()
                p = p0.copy()
                counts = 0
    l_num = len(l_id)
    l_id = np.array(list(set([tuple(i) for i in l_id])))
    return l_id, len(l_id), l_num


# complete IDI generation of EIDIG
def individual_discrimination_generation(dataset_name, config, model):
    # logger Info
    start_time = time.time()
    logger = InfoLogger()

    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']

    num_attrs = len(x[0])
    all_id = np.empty(shape=(0, num_attrs))
    np.random.shuffle(x)

    g_id, g_nd_num, g_num = global_generation(x, num_attrs, config.protected_attrs, config.constraint, model, decay=0.5,
                                              max_iter=10, s_g=1.0)
    l_id, l_nd_num, l_num = local_generation(num_attrs, g_id, config.protected_attrs, config.constraint, model,
                                             update_interval=2, l_num=10,  s_l=1.0, epsilon=1e-6)
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






