"""
- This file provides the essential functions for individual discrimination generation.
- For ADF, EIDIG and NeuronFair
"""

import itertools
import numpy as np
import tensorflow as tf
from keras.models import Model


# return all similar inputs which only different from the protected attributes
def get_similar_set(x, num_attrs, protected_attrs, constraint):
    similar_x = np.empty(shape=(0, num_attrs))
    protected_domain = []
    for i in protected_attrs:
        protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1] + 1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for comb in all_combs:
        x_new = x.copy()
        for attr, value in zip(protected_attrs, comb):
            x_new[attr] = value
        similar_x = np.append(similar_x, [x_new], axis=0)
    return similar_x


# Determine whether a pair of instances is discriminate instance pair
def is_discriminatory(x, similar_x_set, model):
    y_predict = (model(tf.constant([x])) > 0.5)
    for x_new in similar_x_set:
        if (model(tf.constant([x_new])) > 0.5) != y_predict:
            return True
    return False


# Select an instance that the DNN outputs maximally different form the original input
def argmax(x, similar_x_set, model):
    y = model(tf.constant([x]))
    max_dist = 0.0
    x_potential = x.copy()
    for x_new in similar_x_set:
        distance = np.sum(np.square(y - model(tf.constant([x_new]))))
        if distance > max_dist:
            max_dist = distance
            x_potential = x_new.copy()
    return x_potential


# clip the generated instance to satisfy th constraint
def clip(instance, constraint):
    return np.minimum(constraint[:, 1], np.maximum(constraint[:, 0], instance))


# randomly pick an element from a probability distribution
def random_pick(probability):
    random_number = np.random.rand()
    current_probability = 0
    for i in range(len(probability)):
        current_probability += probability[i]
        if current_probability > random_number:
            return i


# find a discriminatory pair given an individual discriminatory instance
def find_idi_pair(x, similar_x_set, model):
    pairs = np.empty(shape=(0, len(x)))
    y_predict = (model(tf.constant([x])) > 0.5)
    for x_pair in similar_x_set:
        if (model(tf.constant([x_pair])) > 0.5) != y_predict:
            pairs = np.append(pairs, [x_pair], axis=0)
    selected_index = random_pick([1.0 / pairs.shape[0]] * pairs.shape[0])
    return pairs[selected_index]


# find all discriminate instance pairs
def create_idi_pair(x, similar_x_set, model):
    pairs = np.empty(shape=(0, len(x)))
    y_predict = (model(tf.constant([x])) > 0.5)
    found = False
    pair = None
    for x_new in similar_x_set:
        if (model(tf.constant([x_new])) > 0.5) != y_predict:
            found = True
            pairs = np.append(pairs, [x_new], axis=0)

    if found:
        selected_index = random_pick([1.0 / pairs.shape[0]] * pairs.shape[0])
        pair = pairs[selected_index]
    return found, pair


# gradient normalization during local search
def normalization(grad1, grad2, protected_attrs, epsilon):
    gradient = np.zeros_like(grad1)
    grad1 = np.abs(grad1)
    grad2 = np.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attrs:
            gradient[i] = 0.0
    gradient_sum = np.sum(gradient)
    probability = gradient / gradient_sum
    return probability


# find the most biased layer and biased neurons in the biased layer
def find_biased_layer_and_neurons(x, model, step_interval=0.005):
    x1 = x.copy()
    x2 = x.copy()
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    auc = np.empty((0,))
    biased_neuron_list = []

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    x1_activations = activation_model.predict(x1)[1:-1]
    x2_activations = activation_model.predict(x2)[1:-1]

    for activation1, activation2 in zip(x1_activations, x2_activations):
        n = activation1.T.shape[1]
        z_l = np.sum(abs(activation1.T - activation2.T), axis=1) / n
        z_l = np.tanh(z_l)
        max_z = max(z_l)

        x_tmp = np.arange(0.0, max_z, step_interval)
        y_tmp = np.empty((0,))
        for count in range(len(x_tmp)):
            y_tmp = np.append(y_tmp, [np.sum(z_l > x_tmp[count])], axis=0)
        y_tmp = y_tmp / len(z_l)

        area = 0.0
        for count in range(len(y_tmp)):
            area += y_tmp[count] * step_interval
        auc = np.append(auc, [area], axis=0)

        t_d = 0.0
        atol = 1
        for i in range(len(x_tmp)):
            if np.isclose(y_tmp[i], x_tmp[i], atol=atol):
                t_d = x_tmp[i]
                atol = abs(y_tmp[i] - x_tmp[i])
        biased_neurons = np.array(z_l > t_d).astype(int)
        biased_neuron_list.append(biased_neurons)

    auc_max_index = np.argmax(auc)

    return auc_max_index + 1, biased_neuron_list[auc_max_index]
