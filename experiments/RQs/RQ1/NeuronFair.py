"""
Algorithm NeuronFair.
- RQ1
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /baseline/NeuronFair.py.
"""

import sys
import time
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append('..')
from utils import utils
from experiments.logfile.InfoLogger import InfoLogger


# compute the gradient of NeuronFair
def compute_grad(x1, x2, new_model, neurons, r):
    p = neurons | r

    x1 = tf.constant([x1], dtype=tf.float32)
    x2 = tf.constant([x2], dtype=tf.float32)
    p = tf.constant(p, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x1)
        y_predict_x1 = tf.nn.sigmoid(new_model(x1))
        y_predict_x2 = new_model(x2)
        y_predict = -1 * tf.reduce_sum(p * y_predict_x2 * y_predict_x1)

    gradient = tape.gradient(y_predict, x1)
    return gradient[0].numpy()


# global_generation of NeuronFair
def global_generation(seeds, num_attrs, protected_attrs, constraint, model, new_model, neurons,
                      decay, max_iter, s_g, r_g, p_r):
    g_id = np.empty(shape=(0, num_attrs))
    g_num = len(seeds)
    neuron_num = len(neurons)
    r = np.zeros(neuron_num, dtype=int)
    finish = False

    for i in range(g_num):
        if finish:
            break

        x1 = seeds[i].copy()
        grad1 = np.zeros_like(x1).astype(float)
        grad2 = np.zeros_like(x1).astype(float)

        for _ in range(max_iter):
            if s_g % r_g == 0:
                random_indices = np.random.choice(neuron_num, int(p_r * neuron_num), replace=False)
                r[random_indices] = 1
            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x1, similar_x1_set, model):
                g_id = np.append(g_id, [x1], axis=0)
                if len(g_id) >= 1000:
                    finish = True
                break

            x2 = utils.argmax(x1, similar_x1_set, model)
            grad1 = decay * grad1 + compute_grad(x1, x2, new_model, neurons, r)
            grad2 = decay * grad2 + compute_grad(x2, x1, new_model, neurons, r)
            direction = np.sign(grad1 + grad2)
            for attr in range(num_attrs):
                if attr in protected_attrs:
                    direction[attr] = 0
            x1 = x1 + direction * s_g
            x1 = utils.clip(x1, constraint)
    g_num = len(g_id)
    g_id = np.array(list(set([tuple(i) for i in g_id])))
    return g_id, len(g_id), g_num


# local_generation of NeuronFair
def local_generation(num_attrs, g_id, protected_attrs, constraint, model, new_model, neurons,
                     decay,  l_num, s_l, r_l, p_r, epsilon):
    l_id = np.empty(shape=(0, num_attrs))
    neuron_num = len(neurons)
    r = np.zeros(neuron_num, dtype=int)

    for x1 in g_id:
        grad1 = np.zeros_like(x1).astype(float)
        grad2 = np.zeros_like(x1).astype(float)
        for _ in range(l_num):
            x1_copy = x1.copy()
            if s_l % r_l == 0:
                random_indices = np.random.choice(neuron_num, int(p_r * neuron_num), replace=False)
                r[random_indices] = 1

            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            x2 = utils.find_idi_pair(x1, similar_x1_set, model)
            grad1 = decay * grad1 + compute_grad(x1, x2, new_model, neurons, r)
            grad2 = decay * grad2 + compute_grad(x2, x1, new_model, neurons, r)
            direction = np.sign(grad1 + grad2)
            p_direction = tf.nn.softmax(1.0 / (abs(grad1 + grad2) + epsilon))

            for attr in range(num_attrs):
                if attr not in protected_attrs:
                    p_tmp = np.random.rand()
                    if p_tmp < p_direction[attr]:
                        x1[attr] = x1[attr] + direction[attr] * s_l
            x1 = utils.clip(x1, constraint)
            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x1, similar_x1_set, model):
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x1_copy
    l_num = len(l_id)
    l_id = np.array(list(set([tuple(i) for i in l_id])))

    return l_id, len(l_id), l_num


# complete IDI generation of NeuronFair
def individual_discrimination_generation(dataset_name, config, model):
    # logger Info
    start_time = time.time()
    logger = InfoLogger()

    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']
    layer_index, neurons = utils.find_biased_layer_and_neurons(x, model)

    # construct new model
    new_model_input = keras.Input(model.layers[0].input_shape)
    new_model_output = new_model_input
    for layer in model.layers[:layer_index + 1]:
        new_model_output = layer(new_model_output)
    new_model = keras.Model(inputs=new_model_input, outputs=new_model_output)

    num_attrs = len(x[0])
    all_id = np.empty(shape=(0, num_attrs))
    np.random.shuffle(x)

    g_id, g_nd_num, g_num = global_generation(x, num_attrs, config.protected_attrs, config.constraint, model, new_model,
                                              neurons, decay=0.5, max_iter=10, s_g=1.0, r_g=1.0, p_r=0.05)
    l_id, l_nd_num, l_num = local_generation(num_attrs, g_id, config.protected_attrs, config.constraint, model,
                                             new_model, neurons, decay=0.5, l_num=10, s_l=1.0, r_l=1.0, p_r=0.05,
                                             epsilon=1e-6)
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
