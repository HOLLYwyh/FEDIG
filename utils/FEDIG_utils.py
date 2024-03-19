"""
- The file provides the essential functions for individual discrimination generation.
- For FEDIG
"""
import random

import numpy as np
import tensorflow as tf
from utils import utils
from itertools import product


# sort the features via FDB (feature biased degree)
def sort_biased_features(data, num_attrs, model, protected_attrs, constraint, min_len):
    biased_features_list = []
    sensitive_features_list = []
    attr_list = []

    for attr in range(num_attrs):
        if attr not in protected_attrs:
            attr_list.append(attr)

    data_len = min(len(data), min_len)
    random_indices = random.sample(range(len(data)), data_len)
    data = data[random_indices]

    for x in data:
        biased_features = []
        for attr in attr_list:
            perturbed_features = np.empty(shape=(0, num_attrs))
            for i in range(constraint[attr][0], constraint[attr][1]+1):
                if i != x[attr]:
                    x1 = x.copy()
                    x1[attr] = i
                    perturbed_features = np.append(perturbed_features, [x1], axis=0)

            sensitivity = 0.0
            layer_outputs = [layer.output for layer in model.layers]
            activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

            for perturb_feature in perturbed_features:
                perturb_feature = tf.constant([perturb_feature], dtype=tf.float32)
                outputs_perturb = activation_model(perturb_feature)
                outputs = activation_model(tf.constant([x], dtype=tf.float32))
                for arr1, arr2 in zip(outputs, outputs_perturb):
                    sensitivity += tf.reduce_sum(tf.abs(arr1 - arr2)).numpy()
            sensitivity /= len(perturbed_features)
            biased_features.append(sensitivity)
        sensitive_features_list.append(biased_features)
    sensitive_features_list = np.mean(sensitive_features_list, axis=0)

    for index, sensitivity in zip(attr_list, sensitive_features_list):
        biased_features_list.append((index, sensitivity))

    biased_features_list = sorted(biased_features_list, key=lambda line: line[1], reverse=True)

    return biased_features_list


# spilt the sorted features to irrelevant features and optimal features
def spilt_biased_features(all_biased_features):
    fbd_list = [t[1] for t in all_biased_features]
    l_value = max(np.mean(fbd_list), np.median(fbd_list))
    r_value = min(np.mean(fbd_list), np.median(fbd_list))
    left_t = right_t = 0

    for i in range(len(fbd_list)):
        if fbd_list[i] >= l_value:
            left_t = i
        if fbd_list[i] >= r_value:
            right_t = i
    optimal_value = fbd_list[:left_t + 1]
    irrelevant_value = np.flip(fbd_list[-right_t:])

    # get biased features
    delta_list = []
    for i in range(len(optimal_value)-1):
        delta_list.append(optimal_value[i]-optimal_value[i+1])
    biased_features = [t[0] for t in all_biased_features][:np.argmax(delta_list)+1]

    # get irrelevant features
    ir_value = min(np.mean(irrelevant_value), np.median(irrelevant_value))
    ir_t = np.argmax(irrelevant_value >= ir_value)
    irrelevant_features = [t[0] for t in all_biased_features][-ir_t:]

    return irrelevant_features, biased_features


# Select an instance that closest to the decision boundary
def find_boundary_pair(x, similar_x_set, model):
    min_dist = 1.0
    x_potential = x.copy()
    for x_new in similar_x_set:
        if np.array_equal(x_new, x):
            distance = np.sum(np.square(0.5 - model(tf.constant([x_new]))))
            if distance < min_dist:
                min_dist = distance
                x_potential = x_new.copy()
    return x_potential


# get all the potential x with the optimal features
def get_potential_global_x(x, direction, features, constraint, s_g):
    potential_x_list = []
    combinations = []

    for attr in range(len(direction)):
        if attr in features:
            options = [direction[attr], 2.0 * direction[attr]]
        else:
            options = [direction[attr]]
        combinations.append(options)

    direction_combinations = list(product(*combinations))

    for d in direction_combinations:
        x1 = x.copy()
        x1 = x1 + np.array(d) * s_g
        x1 = utils.clip(x1, constraint)
        potential_x_list.append(x1)
    potential_x_list = np.array(list(set([tuple(i) for i in potential_x_list])))

    return potential_x_list


# get all the potential x with the irrelevant features
def get_potential_local_x(x, grad_sign, feature, irrelevant_features, constraint, s_l):
    direction = np.zeros_like(x).astype(float)
    potential_x_list = []
    combinations = []

    for attr in range(len(direction)):
        if attr in irrelevant_features:
            options = [0, 2 * grad_sign[attr]]
        elif attr == feature:
            options = [0, utils.random_pick([0.5, 0.5])]
        else:
            options = [0]
        combinations.append(options)

    direction_combinations = list(product(*combinations))

    for d in direction_combinations:
        x1 = x.copy()
        x1 = x1 + np.array(d) * s_l
        x1 = utils.clip(x1, constraint)
        if not np.array_equal(x, x1):
            potential_x_list.append(x1)
    potential_x_list = np.array(list(set([tuple(i) for i in potential_x_list])))

    return potential_x_list


# find all discriminate instance list
def create_idi_list(x, similar_x_set, model):
    idi_list = np.empty(shape=(0, len(x)))
    y_predict = (model(tf.constant([x])) > 0.5)
    found = False
    for x_new in similar_x_set:
        if (model(tf.constant([x_new])) > 0.5) != y_predict:
            found = True
            idi_list = np.append(idi_list, [x_new], axis=0)
    return found, idi_list


# gradient normalization during local search
def normalization(grad1, grad2, protected_attrs, irrelevant_features, epsilon):
    gradient = np.zeros_like(grad1)
    grad1 = np.abs(grad1)
    grad2 = np.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attrs:
            gradient[i] = 0.0
        elif i in irrelevant_features:
            gradient[i] = 0.0
    gradient_sum = np.sum(gradient)
    probability = gradient / gradient_sum
    return probability

