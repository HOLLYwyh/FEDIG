"""
    This file evaluates the performance of CNN models after retraining,
    such as precision, recall and F1 score
"""
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split


def load_image(image_path, file_names, dsize):
    images = []
    for file_name in file_names:
        file_path = os.path.join(image_path, file_name)
        img = cv2.resize(cv2.imread(file_path), dsize)
        images.append(img)
    return np.array(images)


def model_evaluation(x_test, y_test, image_path, dsize, model):
    precision_all = 0.0
    recall_all = 0.0
    f1_score_all = 0.0

    batch_size = 64
    n_batch = len(x_test) // batch_size

    for i in tqdm(range(n_batch), desc='Processing data'):
        start = i * batch_size
        end = (i + 1) * batch_size
        x_batch = []
        for j in range(start, end):
            file_path = os.path.join(image_path, x_test[j])
            img = cv2.resize(cv2.imread(file_path), dsize)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x_batch.append(img)
        y_predict = model.predict(np.array(x_batch))
        y_predict = (y_predict > 0.5).astype(int).flatten()
        y_real = y_test[start: end]

        true_positive = np.dot(y_real, y_predict)
        false_positive = np.sum((y_real - y_predict) == -1.0)
        false_negative = np.sum((y_real - y_predict) == 1.0)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * precision * recall / (precision + recall)

        precision_all += precision
        recall_all += recall
        f1_score_all += f1_score

    precision_all /= n_batch
    recall_all /= n_batch
    f1_score_all /= n_batch
    print('The precision rate is ', precision_all, ', the recall rate is ', recall_all, ', and the F1 score is ', f1_score_all)


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

# get train and test datasets
ca_x_train, ca_x_test, ca_y_train, ca_y_test = train_test_split(ca_file_names, ca_labels, test_size=0.2,
                                                                random_state=30)
ff_x_train, ff_x_test, ff_y_train, ff_y_test = train_test_split(ff_file_names, ff_labels, test_size=0.2,
                                                                random_state=30)

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


ca_dsize = (256, 256)
ff_dsize = (224, 224)

# evaluation
# celebA
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_origin_model)
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_5_retrained_model)
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_10_retrained_model)
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_15_retrained_model)
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_20_retrained_model)
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_25_retrained_model)
model_evaluation(ca_x_test, ca_y_test, ca_image_path, ca_dsize, ca_30_retrained_model)

# FairFace
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_origin_model)
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_5_retrained_model)
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_10_retrained_model)
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_15_retrained_model)
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_20_retrained_model)
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_25_retrained_model)
model_evaluation(ff_x_test, ff_y_test, ff_image_path, ff_dsize, ff_30_retrained_model)


"""
    model evaluation
1. celebA:
    - origin:
        - precision rate:   0.954373632665527
        - recall rate:      0.965244384455184
        - F1 score:         0.9590133520676413
    - 5%:
        - precision rate:   0.9261186183544916
        - recall rate:      0.8498978762914491
        - F1 score:         0.8840126684376127
    - 10%:
        - precision rate:   0.9257715092039288
        - recall rate:      0.9295395332709684
        - F1 score:         0.9263102398021157
    - 15%:
        - precision rate:   0.8621999966864359
        - recall rate:      0.9750984363603314
        - F1 score:         0.9137844291684402
    - 20%:
        - precision rate:   0.9445800254305255
        - recall rate:      0.8418974796503411
        - F1 score:         0.8883851253567301
    - 25%:
         - precision rate:  0.9298882631193399
        - recall rate:      0.8659734369159552
        - F1 score:         0.8947400356625825
    - 30%:
        - precision rate:   0.9299013730633972
        - recall rate:      0.5877215926015399
        - F1 score:         0.7151219532814928
2. FairFace:
    - origin:
        - precision rate:   0.9132836640065108
        - recall rate:      0.9216194851341557 
        - F1 score:         0.9161677848411193
    - 5%:
        - precision rate:   0.9141023256517609
        - recall rate:      0.927586486275119
        - F1 score:         0.9194574125981425
    - 10%:
        - precision rate:   0.9142181046845347
        - recall rate:      0.9262124101680124
        - F1 score:         0.9189196608777239
    - 15%:
        - precision rate:   0.914799889433121
        - recall rate:      0.9217580718257551
        - F1 score:         0.9171784832599037
    - 20%:
        - precision rate:   0.9198811026850423
        - recall rate:      0.9197069894371345
        - F1 score:         0.9184544928366812
    - 25%:
        - precision rate:  0.9115479696966425
        - recall rate:     0.9239591509468787
        - F1 score:        0.9161844426989031
    - 30%:
        - precision rate:  0.9166005408470409
        - recall rate:     0.9302216035114736
        - F1 score:        0.921896702795529
"""
