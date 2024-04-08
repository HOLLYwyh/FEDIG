"""
 This file evaluates the discriminatory degree of retrained CNN models.
"""

import os
import cv2
import sys
import logging
import random
import numpy as np
import pandas as pd
from keras.models import load_model

sys.path.append('../')
from utils import FEDIG_img_utils

logging.basicConfig(filename='fairness_evaluation_img_50.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_protected_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return faces


def load_image(image_path, file_name, dsize):
    file_path = os.path.join(image_path, file_name)
    img = cv2.resize(cv2.imread(file_path), dsize)
    return img


def random_generate(img, generate_num, bbox, model):
    gen_shape = (0,) + img.shape
    gen_id = np.empty(shape=gen_shape)
    random.seed(821)
    for i in range(generate_num):
        img_picked = np.random.rand(*img.shape)
        if FEDIG_img_utils.is_discriminatory(img_picked, model, bbox):
            gen_id = np.append(gen_id, [img_picked], axis=0)
    return gen_id


def evaluate_discrimination(sample_round, generate_num, img, bbox, model):
    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        gen_id = random_generate(img, generate_num, bbox, model)
        percentage = len(gen_id) / generation_num
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)
    logging.info(f'The percentage of individual discriminatory instances with .95 confidence:{avg} ±, {interval}')


# load the dataset
ca_image_path = '../datasets/celebA/img_align_celeba/'
ca_label_path = '../datasets/celebA/Anno/list_attr_celeba.txt'
ff_image_path = '../datasets/FairFace/train/'
ff_label_path = '../datasets/FairFace/'

# labels
ca_labels = []
with open(ca_label_path, 'r') as file:
    file.readline()
    file.readline()
    lines = file.readlines()
    for idx, line in enumerate(lines):
        data = [x for x in line.strip().split(' ') if x != ''][1:]
        ca_labels.append(data[20])
ca_labels = np.array([0 if val == '-1' else 1 for val in ca_labels])

ff_df = pd.read_csv(ff_label_path + 'fairface_label_train.csv', sep=',')
ff_df['gender'] = pd.factorize(ff_df['gender'])[0]
ff_labels = ff_df['gender'].values

# images
ca_file_names = os.listdir(ca_image_path)
ff_file_names = np.array(sorted(os.listdir(ff_image_path), key=lambda x: int(x.split('.')[0])))

# load models
# origin models
ca_origin_path = '../models/trained_models/celeba_model.h5'
ff_origin_path = '../models/trained_models/FairFace_model.h5'

ca_origin_model = load_model(ca_origin_path)
ff_origin_model = load_model(ff_origin_path)

# retrained models
# celebA
ca_5_retrained_path = '../models/retrained_models/celebA_5_retrained_model.h5'
ca_10_retrained_path = '../models/retrained_models/celebA_10_retrained_model.h5'
ca_15_retrained_path = '../models/retrained_models/celebA_15_retrained_model.h5'
ca_20_retrained_path = '../models/retrained_models/celebA_20_retrained_model.h5'
ca_25_retrained_path = '../models/retrained_models/celebA_25_retrained_model.h5'
ca_30_retrained_path = '../models/retrained_models/celebA_30_retrained_model.h5'

# FairFace
ff_5_retrained_path = '../models/retrained_models/FairFace_5_retrained_model.h5'
ff_10_retrained_path = '../models/retrained_models/FairFace_10_retrained_model.h5'
ff_15_retrained_path = '../models/retrained_models/FairFace_15_retrained_model.h5'
ff_20_retrained_path = '../models/retrained_models/FairFace_20_retrained_model.h5'
ff_25_retrained_path = '../models/retrained_models/FairFace_25_retrained_model.h5'
ff_30_retrained_path = '../models/retrained_models/FairFace_30_retrained_model.h5'

ca_5_retrained_model = load_model(ca_5_retrained_path)
ca_10_retrained_model = load_model(ca_10_retrained_path)
ca_15_retrained_model = load_model(ca_15_retrained_path)
ca_20_retrained_model = load_model(ca_20_retrained_path)
ca_25_retrained_model = load_model(ca_25_retrained_path)
ca_30_retrained_model = load_model(ca_30_retrained_path)

ff_5_retrained_model = load_model(ff_5_retrained_path)
ff_10_retrained_model = load_model(ff_10_retrained_path)
ff_15_retrained_model = load_model(ff_15_retrained_path)
ff_20_retrained_model = load_model(ff_20_retrained_path)
ff_25_retrained_model = load_model(ff_25_retrained_path)
ff_30_retrained_model = load_model(ff_30_retrained_path)


# evaluate retrained models
round_num = 100
generation_num = 100
ca_dsize = (256, 256)
ff_dsize = (224, 224)

ca_bbox = []
ff_bbox = []
ca_img = []
ff_img = []

while len(ca_bbox) == 0:
    ca_index = np.random.choice(len(ca_file_names))
    ca_file_name = ca_file_names[ca_index]
    ca_img = load_image(ca_image_path, ca_file_name, ca_dsize)
    ca_bbox = get_protected_features(ca_img)

while len(ff_bbox) == 0:
    ff_index = np.random.choice(len(ff_file_names))
    ff_file_name = ff_file_names[ff_index]
    ff_img = load_image(ff_image_path, ff_file_name, ff_dsize)
    ff_bbox = get_protected_features(ff_img)


# evaluation
logging.info("=============ca=============")
logging.info("=============0=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_origin_model)
logging.info("=============5=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_5_retrained_model)
logging.info("=============10=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_10_retrained_model)
logging.info("=============15=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_15_retrained_model)
logging.info("=============20=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_20_retrained_model)
logging.info("=============25=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_25_retrained_model)
logging.info("=============30=============")
evaluate_discrimination(round_num, generation_num, ca_img, ca_bbox, ca_30_retrained_model)

logging.info("=============ca=============")
logging.info("=============0=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_origin_model)
logging.info("=============5=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_5_retrained_model)
logging.info("=============10=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_10_retrained_model)
logging.info("=============15=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_15_retrained_model)
logging.info("=============20=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_20_retrained_model)
logging.info("=============25=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_25_retrained_model)
logging.info("=============30=============")
evaluate_discrimination(round_num, generation_num, ff_img, ff_bbox, ff_30_retrained_model)


"""
    .95 confidence
1. celebA:
    - original: 0.7603999999999997 ± 0.019026825559719622
    - 5%:       0.24000000000000005 ± 0.06320810074666063
    - 10%:      0.5699999999999999 ± 0.0396869751933805
    - 15%:      0.13799999999999998 ± 0.012309049678996343
    - 20%:      0.4628 ± 0.022091834625490026
    - 25%:      0.7164000000000001 ± 0.014903837937927263
    - 30%:      0.19519999999999998 ± 0.0160886274964647
2. FairFace:
    - original: 0.19879999999999998 ± 0.01657228390777807
    - 5%:       0.0042 ± 0.04458730761102312
    - 10%:      0.0013 ± 0.053046884922679484
    - 15%:      0.0033 ± 0.023268551738344183
    - 20%:      0.0021 ± 0.0619418921247971
    - 25%:      0.0032 ± 0.0023151242558445964
    - 30%:      0.0011 ± 0.014535749034707501
"""
