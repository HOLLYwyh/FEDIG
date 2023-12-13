"""
    This file evaluates the performance of models after retraining,
    such as precision, recall and F1 score
"""

import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


def model_evaluation(y_real, y_predict):
    true_positive = np.dot(y_real, y_predict)
    false_positive = np.sum((y_real - y_predict) == -1.0)
    false_negative = np.sum((y_real - y_predict) == 1.0)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)

    print('The precision rate is ', precision, ', the recall rate is ', recall, ', and the F1 score is ', f1_score)


# load models
# origin models
credit_origin_path = '../models/trained_models/credit_model.h5'
bank_origin_path = '../models/trained_models/bank_model.h5'
census_origin_path = '../models/trained_models/census_model.h5'

credit_origin_model = load_model(credit_origin_path)
bank_origin_model = load_model(bank_origin_path)
census_origin_model = load_model(census_origin_path)

# retrained models
# credit
credit_5_retrained_path = '../models/retrained_models/credit_5_retrained_model.h5'
credit_10_retrained_path = '../models/retrained_models/credit_10_retrained_model.h5'
credit_15_retrained_path = '../models/retrained_models/credit_15_retrained_model.h5'
credit_20_retrained_path = '../models/retrained_models/credit_20_retrained_model.h5'
credit_25_retrained_path = '../models/retrained_models/credit_25_retrained_model.h5'
credit_30_retrained_path = '../models/retrained_models/credit_30_retrained_model.h5'
# bank
bank_5_retrained_path = '../models/retrained_models/bank_5_retrained_model.h5'
bank_10_retrained_path = '../models/retrained_models/bank_10_retrained_model.h5'
bank_15_retrained_path = '../models/retrained_models/bank_15_retrained_model.h5'
bank_20_retrained_path = '../models/retrained_models/bank_20_retrained_model.h5'
bank_25_retrained_path = '../models/retrained_models/bank_25_retrained_model.h5'
bank_30_retrained_path = '../models/retrained_models/bank_30_retrained_model.h5'
# census
census_5_retrained_path = '../models/retrained_models/census_5_retrained_model.h5'
census_10_retrained_path = '../models/retrained_models/census_10_retrained_model.h5'
census_15_retrained_path = '../models/retrained_models/census_15_retrained_model.h5'
census_20_retrained_path = '../models/retrained_models/census_20_retrained_model.h5'
census_25_retrained_path = '../models/retrained_models/census_25_retrained_model.h5'
census_30_retrained_path = '../models/retrained_models/census_30_retrained_model.h5'

credit_5_retrained_model = load_model(credit_5_retrained_path)
credit_10_retrained_model = load_model(credit_10_retrained_path)
credit_15_retrained_model = load_model(credit_15_retrained_path)
credit_20_retrained_model = load_model(credit_20_retrained_path)
credit_25_retrained_model = load_model(credit_25_retrained_path)
credit_30_retrained_model = load_model(credit_30_retrained_path)

bank_5_retrained_model = load_model(bank_5_retrained_path)
bank_10_retrained_model = load_model(bank_10_retrained_path)
bank_15_retrained_model = load_model(bank_15_retrained_path)
bank_20_retrained_model = load_model(bank_20_retrained_path)
bank_25_retrained_model = load_model(bank_25_retrained_path)
bank_30_retrained_model = load_model(bank_30_retrained_path)

census_5_retrained_model = load_model(census_5_retrained_path)
census_10_retrained_model = load_model(census_10_retrained_path)
census_15_retrained_model = load_model(census_15_retrained_path)
census_20_retrained_model = load_model(census_20_retrained_path)
census_25_retrained_model = load_model(census_25_retrained_path)
census_30_retrained_model = load_model(census_30_retrained_path)

# load the datasets:
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

credit_x_train, credit_x_test, credit_y_train, credit_y_test = train_test_split(credit_x, credit_y, test_size=0.4,
                                                                                random_state=821)
bank_x_train, bank_x_test, bank_y_train, bank_y_test = train_test_split(bank_x, bank_y, test_size=0.2,
                                                                        random_state=821)
census_x_train, census_x_test, census_y_train, census_y_test = train_test_split(census_x, census_y, test_size=0.2,
                                                                                random_state=821)

# model evaluation
# credit
model_evaluation(credit_y_test, (credit_origin_model.predict(credit_x_test) > 0.5).astype(int).flatten())
model_evaluation(credit_y_test, (credit_5_retrained_model.predict(credit_x_test) > 0.5).astype(int).flatten())
model_evaluation(credit_y_test, (credit_10_retrained_model.predict(credit_x_test) > 0.5).astype(int).flatten())
model_evaluation(credit_y_test, (credit_15_retrained_model.predict(credit_x_test) > 0.5).astype(int).flatten())
model_evaluation(credit_y_test, (credit_20_retrained_model.predict(credit_x_test) > 0.5).astype(int).flatten())
model_evaluation(credit_y_test, (credit_25_retrained_model.predict(credit_x_test) > 0.5).astype(int).flatten())
model_evaluation(credit_y_test, (credit_30_retrained_model.predict(credit_x_test) > 0.5).astype(int).flatten())

# bank
model_evaluation(bank_y_test, (bank_origin_model.predict(bank_x_test) > 0.5).astype(int).flatten())
model_evaluation(bank_y_test, (bank_5_retrained_model.predict(bank_x_test) > 0.5).astype(int).flatten())
model_evaluation(bank_y_test, (bank_10_retrained_model.predict(bank_x_test) > 0.5).astype(int).flatten())
model_evaluation(bank_y_test, (bank_15_retrained_model.predict(bank_x_test) > 0.5).astype(int).flatten())
model_evaluation(bank_y_test, (bank_20_retrained_model.predict(bank_x_test) > 0.5).astype(int).flatten())
model_evaluation(bank_y_test, (bank_25_retrained_model.predict(bank_x_test) > 0.5).astype(int).flatten())
model_evaluation(bank_y_test, (bank_30_retrained_model.predict(bank_x_test) > 0.5).astype(int).flatten())


# census
model_evaluation(census_y_test, (census_origin_model.predict(census_x_test) > 0.5).astype(int).flatten())
model_evaluation(census_y_test, (census_5_retrained_model.predict(census_x_test) > 0.5).astype(int).flatten())
model_evaluation(census_y_test, (census_10_retrained_model.predict(census_x_test) > 0.5).astype(int).flatten())
model_evaluation(census_y_test, (census_15_retrained_model.predict(census_x_test) > 0.5).astype(int).flatten())
model_evaluation(census_y_test, (census_20_retrained_model.predict(census_x_test) > 0.5).astype(int).flatten())
model_evaluation(census_y_test, (census_25_retrained_model.predict(census_x_test) > 0.5).astype(int).flatten())
model_evaluation(census_y_test, (census_30_retrained_model.predict(census_x_test) > 0.5).astype(int).flatten())

"""
    model evaluation
1. credit:
    - origin:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 5%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 10%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 15%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 20%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 25%:
         - precision rate:
        - recall rate: 
        - F1 score: 
    - 30%:
        - precision rate:
        - recall rate: 
        - F1 score: 
2. bank:
    - origin:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 5%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 10%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 15%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 20%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 25%:
         - precision rate:
        - recall rate: 
        - F1 score: 
    - 30%:
        - precision rate:
        - recall rate: 
        - F1 score: 
3. census:
    - origin:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 5%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 10%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 15%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 20%:
        - precision rate:
        - recall rate: 
        - F1 score: 
    - 25%:
         - precision rate:
        - recall rate: 
        - F1 score: 
    - 30%:
        - precision rate:
        - recall rate: 
        - F1 score: 
"""