"""
Algorithm FEDIG.
- RQ4
- For experimental perspective, we add some print-statement to observe the inside details.
- If you focus on the algorithm itself, please refer to the version without print-statement at /FEDIG/FEDIG_img.py.
"""

import os
import sys
import cv2
import time
import logging
import numpy as np
import tensorflow as tf

sys.path.append('..')
from utils import FEDIG_img_utils
from experiments.logfile.InfoLogger import InfoLogger


# load image
def load_image(image_path, file_name, dsize):
    file_path = os.path.join(image_path, file_name)
    img = cv2.resize(cv2.imread(file_path), dsize)
    return img


# compute the gradient of model predictions w.r.t. input attributes
def compute_grad(x, model):
    x = tf.constant([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_predict = model(x)
    gradient = tape.gradient(y_predict, x)
    return gradient[0].numpy()


# classifier for protected features detecting
def get_protected_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return faces


# global generation of FEDIG
def global_generation(img, img_shape, model, bbox, biased_features, decay, s_g):
    g_id = np.empty(shape=img_shape)

    img0 = img.copy()
    grad1 = np.zeros_like(img0).astype(float)
    grad2 = np.zeros_like(img0).astype(float)
    x, y, w, h = bbox[0]

    if FEDIG_img_utils.is_discriminatory(img0, model, bbox):
        g_id = np.append(g_id, [img0], axis=0)

    img2 = FEDIG_img_utils.find_idi_pair(img0, model, bbox)
    if not np.array_equal(img0, img2):
        grad1 = decay * grad1 + compute_grad(img0, model)
        grad2 = decay * grad2 + compute_grad(img2, model)
        direction = np.sign(grad1 + grad2)
        direction[y:y + h, x:x + w] = 0
        potential_img_list = FEDIG_img_utils.get_potential_global_img(img0, biased_features, direction, s_g)
        for img1 in potential_img_list:
            if FEDIG_img_utils.is_discriminatory(img1, model, bbox):
                g_id = np.append(g_id, [img1], axis=0)

    return g_id


# local generation of FEDIG
def local_generation(g_id, img_shape, model, bbox, irrelevant_features, decay, s_l):
    l_id = np.empty(shape=img_shape)
    x, y, w, h = bbox[0]

    for img1 in g_id:
        grad1 = np.zeros_like(img1).astype(float)
        grad2 = np.zeros_like(img1).astype(float)

        img2 = FEDIG_img_utils.find_idi_pair(img1, model, bbox)
        if not np.array_equal(img1, img2):
            grad1 = decay * grad1 + compute_grad(img1, model)
            grad2 = decay * grad2 + compute_grad(img2, model)

            direction = np.sign(grad1 + grad2)
            direction[y:y + h, x:x + w] = 0

            p = FEDIG_img_utils.normalization(grad1, grad2, bbox, irrelevant_features, 1e-6)
            feature = FEDIG_img_utils.random_pick(p)
            potential_img_list = FEDIG_img_utils.get_potential_local_img(img1, feature, irrelevant_features, direction,
                                                                         s_l)

            for img in potential_img_list:
                if FEDIG_img_utils.is_discriminatory(img, model, bbox):
                    l_id = np.append(l_id, [img], axis=0)

    return l_id


# complete IDI generation of FEDIG
def cnn_idi_generation(file_names, model, image_path, dsize, decay=0.5):
    logging.info('Start................')
    # logger Info
    start_time = time.time()
    logger = InfoLogger()

    img_shape = (0,) + load_image(image_path, file_names[0], dsize).shape
    all_id = np.empty(shape=img_shape)
    g_all_id = np.empty(shape=img_shape)
    l_all_id = np.empty(shape=img_shape)

    num = 0
    for file_name in file_names:
        num += 1
        print(f"Current file_num: {num}...........")
        logging.info(f"Current file_num: {num}...........")
        print(f"Current generated num: {len(all_id)}...........")
        logging.info(f"Current generated num: {len(all_id)}...........")

        img = load_image(image_path, file_name, dsize)
        bbox = get_protected_features(img)
        if len(bbox) == 0:
            num -= 1
            continue
        if num == 30:
            break
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # calculate biased features and irrelevant features
        all_biased_features = FEDIG_img_utils.sort_img_biased_features(img, model, bbox)
        irrelevant_features, biased_features = FEDIG_img_utils.spilt_img_biased_features(all_biased_features)

        g_id = global_generation(img, img_shape, model, bbox, biased_features, decay, s_g=0.1)
        l_id = local_generation(g_id, img_shape, model, bbox, irrelevant_features, decay, s_l=0.001)

        part_id = np.append(g_id, l_id, axis=0)
        all_id = np.append(all_id, part_id, axis=0)

        g_all_id = np.append(g_all_id, g_id, axis=0)
        l_all_id = np.append(l_all_id, l_id, axis=0)

    all_id = np.unique(all_id, axis=0)

    logger.set_all_non_duplicate_number(len(all_id))
    logger.set_all_number(len(g_all_id) + len(l_all_id))
    logger.set_global_number(len(g_all_id))
    logger.set_global_non_duplicate_number(len(np.unique(g_all_id, axis=0)))
    logger.set_local_number(len(l_all_id))
    logger.set_local_non_duplicate_number(len(np.unique(l_all_id, axis=0)))
    logger.set_total_time(time.time() - start_time)

    return logger
