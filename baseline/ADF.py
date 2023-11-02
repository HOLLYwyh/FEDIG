"""
This file will reproduce the algorithm ADF, one state-of-art individual discrimination generation algorithm.
The source code of ADF can be accessed at https://github.com/pxzhang94/ADF
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
sys.path.append('..')
from utils import utils


# compute the gradient of loss function w.r.t input attributes
def compute_grad(x, model, loss_func=keras.losses.binary_crossentropy):
    x = tf.constant([x], dtype=tf.float32)
    y_predict = tf.cast(model(x) > 0.5, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(y_predict, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()


# global_generation of ADF
def global_generation(x, seeds, num_attrs, protected_attrs, constraint, model, max_iter, s_g):
    g_id = np.empty(shape=(0, num_attrs))
    g_num = len(seeds)

    for i in range(g_num):
        x1 = seeds[i].copy()
        for _ in range(max_iter):
            similar_x1_set = utils.get_similar_set(x1, num_attrs, protected_attrs, constraint)
            if utils.is_discriminatory(x1, similar_x1_set, model):
                g_id = np.append(g_id, [x1], axis=0)
                break
            x2 = utils.argmax(x1, similar_x1_set, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction = np.zeros_like(x[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attr in range(num_attrs):
                if attr not in protected_attrs and sign_grad1[attr] == sign_grad2[attr]:
                    direction[attr] = sign_grad1[attr]
            x1 = x1 + s_g * direction
            x1 = utils.clip(x1, constraint)
    g_id = np.array(list(set([tuple(i) for i in g_id])))
    return g_id


# local_generation of ADF
def local_generation(num_attrs, l_num, g_id, protected_attrs, constraint, model, s_l, epsilon):
    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attrs))


# IDI generation
def individual_discrimination_generation():
    pass
