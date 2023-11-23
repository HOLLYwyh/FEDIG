"""
This file will reproduce the algorithm NeuronFair, one state-of-art individual discrimination generation algorithm.
The source code of NeuronFair can be accessed at https://github.com/haibinzheng/NeuronFair
"""

import sys
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append('..')
from utils import utils


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

    for i in range(g_num):
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
    g_id = np.array(list(set([tuple(i) for i in g_id])))
    return g_id


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
    l_id = np.array(list(set([tuple(i) for i in l_id])))
    return l_id


# complete IDI generation of NeuronFair
def individual_discrimination_generation(dataset_name, config, model, c_num=4):
    data_path = '../clusters/' + dataset_name + '.pkl'
    cluster_data = joblib.load(data_path)
    x = cluster_data['X']
    labels = cluster_data['cluster_labels']
    layer_index, neurons = utils.find_biased_layer_and_neurons(x, model)

    # construct new model
    new_model_input = keras.Input(model.layers[0].input_shape)
    new_model_output = new_model_input
    for layer in model.layers[:layer_index + 1]:
        new_model_output = layer(new_model_output)
    new_model = keras.Model(inputs=new_model_input, outputs=new_model_output)

    num_attrs = len(x[0])
    all_id = np.empty(shape=(0, num_attrs))
    clusters = [[] for _ in range(c_num)]
    for i, label in enumerate(labels):
        clusters[label].append(x[i])

    for i in range(len(clusters)):
        g_id = global_generation(clusters[i], num_attrs, config.protected_attrs, config.constraint, model, new_model,
                                 neurons, decay=0.5, max_iter=10, s_g=1.0, r_g=1.0, p_r=0.05)
        l_id = local_generation(num_attrs, g_id, config.protected_attrs, config.constraint, model, new_model,
                                neurons, decay=0.5, l_num=10, s_l=1.0, r_l=1.0, p_r=0.05, epsilon=1e-6)
        part_id = np.append(g_id, l_id, axis=0)
        all_id = np.append(all_id, part_id, axis=0)

    all_id = np.array(list(set([tuple(i) for i in all_id])))
    return all_id
