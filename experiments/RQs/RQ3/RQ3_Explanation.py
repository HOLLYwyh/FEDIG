"""
This file is the experiments of RQ3:
    - Explanation of the unfairness with features
"""

import sys
import pandas as pd
import FEDIG
import NeuronFair

from keras.models import load_model

sys.path.append('..')
from utils import config

# model path
credit_model_path = '../models/trained_models/credit_model.h5'
census_model_path = '../models/trained_models/census_model.h5'
bank_model_path = '../models/trained_models/bank_model.h5'

# load models and datasets
credit_model = load_model(credit_model_path)
census_model = load_model(census_model_path)
bank_model = load_model(bank_model_path)

# save the logging data
file_path = 'logfile/RQs/RQ3.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['explanation', 'time', 'layer'])


# features of dataset
credit_features = ['BalanceCheque', 'Loan_NurnMonth', 'CreditHistory', 'CreditAmt', 'SavingsBalance', 'Mths_employ',
                   'PersonStatusSex', 'PresentResidence', 'Property', 'AgeInYears', 'OtherInstPlans',
                   'NumCreditsThisBank', 'NumPplLiablMaint', 'Telephone', 'ForeignWorker', 'Purpose_CarNew',
                   'Purpose_CarOld', 'otherdebtor_noneVsGuar', 'otherdebt_coapplVsGuar', 'house_rentVsFree',
                   'house_ownsVsFree', 'job_unemployedVsMgt', 'jobs_unskilledVsMgt', 'job_skilledVsMgt']
bank_features = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                 "month", "duration", "campaign", "pdays", "previous", "poutcome"]
census_features = ["age", "workclass", "fnlwgt", "education-num", "marital-status", "occupation", "relationship",
                   "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

# FEDIG
credit_biased_list, credit_time = FEDIG.find_biased_features('credit', config.Credit, credit_model, 2)
credit_biased_features = []
for i, value in credit_biased_list:
    credit_biased_features.append(credit_features[i])

bank_biased_list, bank_time = FEDIG.find_biased_features('bank', config.Bank, bank_model, 2)
bank_biased_features = []
for i, value in bank_biased_list:
    bank_biased_features.append(bank_features[i])

census_biased_list, census_time = FEDIG.find_biased_features('census', config.Census, census_model, 2)
census_biased_features = []
for i, value in census_biased_list:
    census_biased_features.append(census_features[i])

credit_biased_features = credit_biased_features[: int(0.1 * len(credit_biased_features))]
bank_biased_features = bank_biased_features[: int(0.15 * len(bank_biased_features))]
census_biased_features = census_biased_features[: int(0.1 * len(census_biased_features))]

df = df.append(pd.DataFrame({'explanation': ['FEDIG'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': ['credit'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': [credit_biased_features], 'time': [credit_time], 'layer': ['-']}, index=[0]))

df = df.append(pd.DataFrame({'explanation': ['bank'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': [bank_biased_features], 'time': [bank_time], 'layer': ['-']}, index=[0]))

df = df.append(pd.DataFrame({'explanation': ['census'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': [census_biased_features], 'time': [census_time], 'layer': ['-']}, index=[0]))


# Neuron Fair
credit_biased_neurons, credit_nf_time, credit_nf_layer = NeuronFair.find_biased_neurons('credit', credit_model)
bank_biased_neurons, bank_nf_time, bank_nf_layer = NeuronFair.find_biased_neurons('bank', bank_model)
census_biased_neurons, census_nf_time, census_nf_layer = NeuronFair.find_biased_neurons('census', census_model)

df = df.append(pd.DataFrame({'explanation': ['NeuronFair'], 'time': ['-'], 'layer': ['-']}, index=[0]),
               ignore_index=True)
df = df.append(pd.DataFrame({'explanation': ['credit'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': [credit_biased_neurons], 'time': [credit_nf_time],
                             'layer': [credit_nf_layer]}, index=[0]))

df = df.append(pd.DataFrame({'explanation': ['bank'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': [bank_biased_neurons], 'time': [bank_nf_time], 'layer': [bank_nf_layer]},
                            index=[0]))

df = df.append(pd.DataFrame({'explanation': ['census'], 'time': ['-'], 'layer': ['-']}, index=[0]), ignore_index=True)
df = df.append(pd.DataFrame({'explanation': [census_biased_neurons], 'time': [census_nf_time], 'layer': [census_nf_layer]},
                            index=[0]))

df.to_csv(file_path, index=False)
