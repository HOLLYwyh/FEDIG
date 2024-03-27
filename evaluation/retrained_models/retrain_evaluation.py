"""
    This file evaluates the performance of models after retraining,
    such as precision, recall and F1 score
"""


import numpy as np
import pandas as pd
from keras.models import load_model
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
        - precision rate:   0.6060606060606061
        - recall rate:      0.5660377358490566
        - F1 score:         0.5853658536585366
    - 5%:
        - precision rate:   0.625
        - recall rate:      0.471698113207547
        - F1 score:         0.5376344086021505
    - 10%:
        - precision rate:   0.6206896551724138
        - recall rate:      0.5094339622641509
        - F1 score:         0.5595854922279793
    - 15%:
        - precision rate:   0.6716417910447762
        - recall rate:      0.42452830188679247
        - F1 score:         0.5202312138728324
    - 20%:
        - precision rate:   0.6417910447761194
        - recall rate:      0.4056603773584906
        - F1 score:         0.4971098265895953
    - 25%:
         - precision rate:  0.6388888888888888
        - recall rate:      0.4339622641509434
        - F1 score:         0.5168539325842696
    - 30%:
        - precision rate:   0.6428571428571429
        - recall rate:      0.5094339622641509 
        - F1 score:         0.568421052631579
2. bank:
    - origin:
        - precision rate:   0.6758409785932722
        - recall rate:      0.21709233791748528
        - F1 score:         0.3286245353159852
    - 5%:
        - precision rate:   0.7015873015873015 
        - recall rate:      0.21709233791748528
        - F1 score:         0.33158289572393096
    - 10%:
        - precision rate:   0.6656976744186046
        - recall rate:      0.224950884086444
        - F1 score:         0.33627019089574156
    - 15%:
        - precision rate:   0.6805111821086262
        - recall rate:      0.20923379174852652
        - F1 score:         0.32006010518407213
    - 20%:
        - precision rate:   0.6512261580381471
        - recall rate:      0.23477406679764243
        - F1 score:         0.3451263537906137
    - 25%:
         - precision rate:  0.6722689075630253
        - recall rate:      0.2357563850687623
        - F1 score:         0.3490909090909091
    - 30%:
        - precision rate:   0.7049180327868853
        - recall rate:      0.2111984282907662
        - F1 score:         0.3250188964474679
3. census:
    - origin:
        - precision rate:   0.6715145436308927
        - recall rate:      0.5668924640135479
        - F1 score:         0.6147842056932966
    - 5%:
        - precision rate:   0.6887966804979253
        - recall rate:      0.5622353937341237
        - F1 score:         0.6191142191142192
    - 10%:
        - precision rate:   0.6955380577427821
        - recall rate:      0.560965283657917
        - F1 score:         0.6210452308413404
    - 15%:
        - precision rate:   0.7139588100686499
        - recall rate:      0.5283657917019475
        - F1 score:         0.6072992700729928
    - 20%:
        - precision rate:   0.697560975609756
        - recall rate:      0.5448772226926334
        - F1 score:         0.6118374138340861
    - 25%:
         - precision rate:  0.7320424734540912
        - recall rate:      0.4961896697713802
        - F1 score:         0.5914711077466566
    - 30%:
        - precision rate:   0.6968085106382979
        - recall rate:      0.554614733276884
        - F1 score:         0.6176331918906177
"""