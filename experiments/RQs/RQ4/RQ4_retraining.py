"""
- RQ4
- This file retrains the models with individual discriminatory instances.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

sys.path.append('..')
from utils import config


# we randomly sample 5% of individual discriminatory instances generated by FIDIG for retraining
def retraining(dataset_name, config, model, x_train, x_test, y_train, y_test, idi, model_name, idi_len):
    classifier = joblib.load('../models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    protected_features = config.protected_attrs

    selected_indices = np.random.choice(idi.shape[0], size=idi_len, replace=False)
    idi_selected = idi[selected_indices, :]

    vote_label = classifier.predict(np.delete(idi_selected, protected_features, axis=1))
    x_train = np.append(x_train, idi_selected, axis=0)
    y_train = np.append(y_train, vote_label, axis=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=821)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.evaluate(x_test, y_test)
    model.save('../models/retrained_models/' + dataset_name + '_' + model_name + '_retrained_model.h5')


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

# spilt the training set and test set
credit_x_train, credit_x_test, credit_y_train, credit_y_test = train_test_split(credit_x, credit_y, test_size=0.4,
                                                                                random_state=821)
bank_x_train, bank_x_test, bank_y_train, bank_y_test = train_test_split(bank_x, bank_y, test_size=0.2,
                                                                        random_state=821)
census_x_train, census_x_test, census_y_train, census_y_test = train_test_split(census_x, census_y, test_size=0.2,
                                                                                random_state=821)
# load models
credit_model_path = '../models/trained_models/credit_model.h5'
census_model_path = '../models/trained_models/census_model.h5'
bank_model_path = '../models/trained_models/bank_model.h5'

credit_model = load_model(credit_model_path)
census_model = load_model(census_model_path)
bank_model = load_model(bank_model_path)


# retrain begins
# credit
# credit_idi = np.load('./logfile/generated_instances/credit_discriminatory_instance.npy')
# # 5%
# retraining('credit', config.Credit, credit_model, credit_x_train, credit_x_test, credit_y_train, credit_y_test,
#            credit_idi, '5', 32)
# # 10%
# retraining('credit', config.Credit, credit_model, credit_x_train, credit_x_test, credit_y_train, credit_y_test,
#            credit_idi, '10', 67)
# 15%
# retraining('credit', config.Credit, credit_model, credit_x_train, credit_x_test, credit_y_train, credit_y_test,
#            credit_idi, '15', 106)
# 20%
# retraining('credit', config.Credit, credit_model, credit_x_train, credit_x_test, credit_y_train, credit_y_test,
#            credit_idi, '20', 150)
# # 25%
# retraining('credit', config.Credit, credit_model, credit_x_train, credit_x_test, credit_y_train, credit_y_test,
#            credit_idi, '25', 200)
# # 30%
# retraining('credit', config.Credit, credit_model, credit_x_train, credit_x_test, credit_y_train, credit_y_test,
#            credit_idi, '30', 257)

# # bank
# bank_idi = np.load('./logfile/generated_instances/bank_discriminatory_instance.npy')
# # 5%
# retraining('bank', config.Bank, bank_model, bank_x_train, bank_x_test, bank_y_train, bank_y_test, bank_idi, '5',
#            1903)
# # 10%
# retraining('bank', config.Bank, bank_model, bank_x_train, bank_x_test, bank_y_train, bank_y_test, bank_idi, '10',
#            4019)
# # 15%
# retraining('bank', config.Bank, bank_model, bank_x_train, bank_x_test, bank_y_train, bank_y_test, bank_idi, '15',
#            6383)
# # 20%
# retraining('bank', config.Bank, bank_model, bank_x_train, bank_x_test, bank_y_train, bank_y_test, bank_idi, '20',
#            9042)
# # 25%
# retraining('bank', config.Bank, bank_model, bank_x_train, bank_x_test, bank_y_train, bank_y_test, bank_idi, '25',
#            12056)
# # 30%
# retraining('bank', config.Bank, bank_model, bank_x_train, bank_x_test, bank_y_train, bank_y_test, bank_idi, '30',
#            15501)

# census
# census_idi = np.load('./logfile/generated_instances/census_discriminatory_instance.npy')
print(len(census_x_train))
# 5%
# retraining('census', config.Census, census_model, census_x_train, census_x_test, census_y_train, census_y_test,
#            census_idi, '5', )
# # 10%
# retraining('census', config.Census, census_model, census_x_train, census_x_test, census_y_train, census_y_test,
#            census_idi, '10', )
# # 15%
# retraining('census', config.Census, census_model, census_x_train, census_x_test, census_y_train, census_y_test,
#            census_idi, '15', )
# # 20%
# retraining('census', config.Census, census_model, census_x_train, census_x_test, census_y_train, census_y_test,
#            census_idi, '20', )
# # 35%
# retraining('census', config.Census, census_model, census_x_train, census_x_test, census_y_train, census_y_test,
#            census_idi, '25', )
# # 30%
# retraining('census', config.Census, census_model, census_x_train, census_x_test, census_y_train, census_y_test,
#            census_idi, '30', )

# retrain ends

"""
1. credit accuracy
    - origin: 0.7550
    - 5%: 0.7850
    - 10%: 0.7875
    - 15%: 0.7925
    - 20%: 0.7825
    - 25%: 0.7850
    - 30%: 0.7950

2. bank accuracy
    - origin: 0.9001
    - 5%: 0.9015
    - 10%: 0.9000
    - 15%: 0.8999
    - 20%: 0.8997
    - 25%: 0.9010
    - 30%: 0.9012

3. census accuracy
    - origin: 0.8282
    - 5%:
    - 10%:
    - 15%:
    - 20%:
    - 25%:
    - 30%:
"""
