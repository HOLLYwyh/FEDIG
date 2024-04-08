"""
This file will implement the algorithm FEDIG for graphs, feature base explainable.
"""

import os
import sys
import cv2
import time

sys.path.append('..')
from utils import FEDIG_img_utils


# load image
def load_image(image_path, file_name, dsize):
    file_path = os.path.join(image_path, file_name)
    img = cv2.resize(cv2.imread(file_path), dsize)
    return img


# classifier for protected features detecting
def get_protected_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return faces


# complete IDI generation of FEDIG
def cnn_idi_generation(file_names, model, image_path, dsize):
    irrelevant_feature_list = []
    biased_feature_list = []
    time_list = []
    total_time = 0

    num = 0
    for file_name in file_names:
        num += 1
        img = load_image(image_path, file_name, dsize)
        bbox = get_protected_features(img)
        if len(bbox) == 0:
            num -= 1
            continue
        if num == 3:
            break

        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # calculate biased features and irrelevant features
        start_time = time.time()
        all_biased_features = FEDIG_img_utils.sort_img_biased_features(img, model, bbox)
        irrelevant_features, biased_features = FEDIG_img_utils.spilt_img_biased_features(all_biased_features)
        end_time = time.time()
        delta_time = end_time - start_time
        time_list.append(delta_time)
        total_time += delta_time
        irrelevant_feature_list.append(irrelevant_features)
        biased_feature_list.append(biased_features)

    return irrelevant_feature_list, biased_feature_list, total_time, time_list
