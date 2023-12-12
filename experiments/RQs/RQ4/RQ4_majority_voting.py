"""
- RQ4
- This file use majority voting for relabeling.
- We use five classifiers are trained for majority voting, including KNN, Naive Bayes, MLP, Random Forest, RBF SVM.
"""

import sys
import joblib
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

sys.path.append('..')
from utils import config

# load the dataset
credit_data_path = '../datasets/credit'
bank_data_path = '../datasets/bank'
census_data_path = '../datasets/census'

credit_data = pd.read_csv(credit_data_path).values
bank_data = pd.read_csv(bank_data_path).values
census_data = pd.read_csv(census_data_path).values

credit_x = credit_data[:, 1:]
credit_y = credit_data[:, 0]
bank_x = bank_data[:, :-1]
bank_y = bank_data[:, -1]
census_x = census_data[:, :-1]
census_y = census_data[:, -1]

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
datasets = [(credit_x, credit_y), (bank_x, bank_y), (census_x, census_y)]
dataset_names = ['credit', 'bank', 'census']

for i, dataset in enumerate(datasets):
    model = clone(classifier)
    x, y = dataset
    # credit
    if i == 0:
        x = np.delete(x, config.Credit.protected_attrs, axis=1)
    # bank
    elif i == 1:
        x = np.delete(x, config.Bank.protected_attrs, axis=1)
    # census
    else:
        x = np.delete(x, config.Census.protected_attrs, axis=1)
    if len(x) > 10000:
        test_size = 0.2
    else:
        test_size = 0.4
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=821)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(dataset_names[i] + ':', score)
    joblib.dump(model, '../models/ensemble_models/' + dataset_names[i] + '_ensemble.pkl')

# credit model: 0.7550
# bank model: 0.8980
# census model: 0.8320






