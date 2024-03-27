"""
- The file provides the essential functions for individual discrimination generation.
- For FEDIG(img)
"""

import numpy as np
import tensorflow as tf


# sort the features via FDB (feature biased degree)
def sort_img_biased_features(img, model, bbox):
    biased_features = []
    x, y, w, h = bbox[0]
    perturb_num = 2
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    feature_list = []
    for j in range(x, x + w + 1):
        feature_list.append((j, y))
        feature_list.append((j, y + h))

    for i in range(y, y + h + 1):
        feature_list.append((x, i))
        feature_list.append((x + w, i))

    for feature in feature_list:
        i = feature[0]
        j = feature[1]
        sensitivity = 0.0
        for factor in range(-int(perturb_num / 2), int(perturb_num / 2) + 1):
            if factor != 0:
                perturb_img = np.copy(img)
                perturb_img[i, j] += 0.1 * factor
                perturb_img = np.clip(perturb_img, 0, 1)
                outputs_perturb = activation_model(tf.constant([perturb_img], dtype=tf.float32))
                outputs = activation_model(tf.constant([img], dtype=tf.float32))
                for arr1, arr2 in zip(outputs, outputs_perturb):
                    sensitivity += tf.reduce_sum(tf.abs(arr1 - arr2)).numpy()
        sensitivity /= perturb_num
        biased_features.append((i, j, sensitivity))

    return sorted(biased_features, key=lambda line: line[2], reverse=True)


# spilt the sorted features to irrelevant features and optimal features
def spilt_img_biased_features(all_biased_features):
    fbd_list = [t[2] for t in all_biased_features]
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
    for i in range(len(optimal_value) - 1):
        delta_list.append(optimal_value[i] - optimal_value[i + 1])
    biased_features = [(t[0], t[1]) for t in all_biased_features][:np.argmax(delta_list) + 1]

    # get irrelevant features
    ir_value = min(np.mean(irrelevant_value), np.median(irrelevant_value))
    ir_t = np.argmax(irrelevant_value >= ir_value)
    irrelevant_features = [(t[0], t[1]) for t in all_biased_features][-ir_t:]

    return irrelevant_features, biased_features


# Determine whether an image is IDI
def is_discriminatory(img, model, bbox):
    x, y, w, h = bbox[0]
    y_predict = (model(tf.constant([img])) > 0.5)
    for i in range(0, 51):
        img_cp = img.copy()
        img_cp[y:y + h, x:x + w] = i * 0.02
        if y_predict != (model(tf.constant([img_cp])) > 0.5):
            return True
    return False


# Find the IDI
def find_idi_pair(img, model, bbox):
    x, y, w, h = bbox[0]
    img_idi = img.copy()
    y_predict = (model(tf.constant([img])) > 0.5)
    for i in range(0, 51):
        img_cp = img.copy()
        img_cp[y:y + h, x:x + w] = i * 0.02
        if y_predict != (model(tf.constant([img_cp])) > 0.5):
            return img_cp
    return img_idi


# get all the potential img with the optimal features
def get_potential_global_img(img, biased_features, direction, s_g):
    potential_img_list = []

    img_cp0 = img.copy()
    img_cp0 += direction * s_g
    img_cp0 = np.clip(img_cp0, 0, 1)
    potential_img_list.append(img_cp0)

    for (x, y) in biased_features:
        img_cp = img_cp0.copy()
        z = np.zeros(shape=direction.shape)
        z[x][y] = direction[x][y] * s_g
        img_cp += z
        img_cp = np.clip(img_cp, 0, 1)
        potential_img_list.append(img_cp)

    return potential_img_list


# get all the potential img with the irrelevant features
def get_potential_local_img(img, feature, irrelevant_features, direction, s_l):
    potential_img_list = []

    img_cp0 = img.copy()
    img_cp0 += direction * s_l
    img_cp0 = np.clip(img_cp0, 0, 1)
    potential_img_list.append(img_cp0)

    for (x, y) in irrelevant_features:
        img_cp = img_cp0.copy()
        z = np.zeros(shape=direction.shape)
        z[x][y] = direction[x][y] * s_l
        img_cp += z
        img_cp = np.clip(img_cp, 0, 1)
        potential_img_list.append(img_cp)

    x, y = feature
    img_cp1 = img_cp0.copy()
    z = np.zeros(shape=direction.shape)
    z[x][y] = direction[x][y] * s_l
    img_cp1 += z
    img_cp1 = np.clip(img_cp1, 0, 1)
    potential_img_list.append(img_cp1)

    return potential_img_list


# gradient normalization during local search
def normalization(grad1, grad2, bbox, irrelevant_features, epsilon):
    gradient = np.zeros_like(grad1)
    grad1 = np.abs(grad1)
    grad2 = np.abs(grad2)
    x, y, w, h = bbox[0]

    for i in range(len(gradient[0])):
        for j in range(len(gradient[1])):
            saliency = grad1[i][j] + grad2[i][j]
            gradient[i][j] = 1.0 / (saliency + epsilon)
            if (i, j) in irrelevant_features:
                gradient[i][j] = 0
            if x <= j < x + w and y <= i < y + h:
                gradient[i][j] = 0

    gradient_sum = np.sum(gradient)
    probability = gradient / gradient_sum

    return probability


# randomly pick an element from a probability distribution
def random_pick(probability):
    random_number = np.random.rand()
    current_probability = 0.0

    for i in range(len(probability[0])):
        for j in range(len(probability[1])):
            current_probability += np.sum(probability[i][j])
            if current_probability > random_number:
                return i, j
