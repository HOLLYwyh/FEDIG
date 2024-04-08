"""
- RQ5
- This file use majority voting for relabeling(img).
- We use five classifiers are trained for majority voting, including KNN, Naive Bayes, MLP, Random Forest, RBF SVM.
"""

import os
import cv2
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='voting.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# load files
def load_image(image_path, file_names, dsize):
    images = []
    for file_name in file_names:
        file_name = os.path.join(image_path, file_name)
        img = cv2.resize(cv2.imread(file_name), dsize)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images.append(img)
    return np.array(images)


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

# select some data
select_num = 1250
ca_dsize = (256, 256)
ff_dsize = (224, 224)
ca_file_names = ca_file_names[:select_num]
ff_file_names = ff_file_names[:select_num]
ca_labels = ca_labels[:select_num]
ff_labels = ff_labels[:select_num]


ca_images = load_image(ca_image_path, ca_file_names, ca_dsize)
ff_images = load_image(ff_image_path, ff_file_names, ff_dsize)


# create classifiers
knn_classifier = KNeighborsClassifier()
mlp_classifier = MLPClassifier(max_iter=500)
nb_classifier = GaussianNB()
rf_classifier = RandomForestClassifier()
svm_classifier = SVC(probability=True)

# majority voting
voting_classifier = VotingClassifier(estimators=[('knn', knn_classifier), ('mlp', mlp_classifier), ('nb', nb_classifier),
                                                 ('rf', rf_classifier), ('svm', svm_classifier)], voting='soft')

classifier = Pipeline([('scaler', StandardScaler()), ('ensemble', voting_classifier)])

# train and save the models
# datasets = [(ca_images, ca_labels), (ff_images, ff_labels)]
# dataset_names = ['celebA', 'FairFace']
datasets = [(ff_images, ff_labels)]
dataset_names = ['FairFace']

for i, dataset in enumerate(datasets):
    model = clone(classifier)
    x, y = dataset
    test_size = 0.2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=821)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(dataset_names[i] + ':', score)
    joblib.dump(model, '../models/ensemble_models/' + dataset_names[i] + '_ensemble.pkl')
    logging.info('Finished...')
    logging.info(f'Score: {score}')


# celeba model: 0.884
# FairFace model: 0.685
