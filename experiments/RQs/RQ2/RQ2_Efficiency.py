"""
This file is the experiments of RQ2:
    - Efficiency of FEDIG
"""

import sys
import pandas as pd
import ADF
import EIDIG
import NeuronFair
import DICE
import FEDIG

from tensorflow.keras.models import load_model

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
file_path = 'logfile/RQs/RQ2.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['data', 'number', 'time', 'global_time', 'local_time'])

# experiment begins
# ADF
print('ADF begins...')
df = df.append(pd.DataFrame({'data': ['ADF'], 'number': ['-'], 'time': ['-'], 'global_time': ['-'],
                             'local_time': ['-']}, index=[0]), ignore_index=True)

ADF_credit_id, ADF_credit_logger = ADF.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'number': [len(ADF_credit_id)], 'time': [ADF_credit_logger.total_time],
                             'global_time': [ADF_credit_logger.global_time], 'local_time': [ADF_credit_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={ADF_credit_logger.total_time}==========={len(ADF_credit_id)}===========")

ADF_bank_id, ADF_bank_logger = ADF.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'number': [len(ADF_bank_id)], 'time': [ADF_bank_logger.total_time],
                             'global_time': [ADF_bank_logger.global_time], 'local_time': [ADF_bank_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={ADF_bank_logger.total_time}==========={len(ADF_bank_id)}===========")

ADF_census_id, ADF_census_logger = ADF.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'number': [len(ADF_census_id)], 'time': [ADF_census_logger.total_time],
                             'global_time': [ADF_census_logger.global_time], 'local_time': [ADF_census_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={ADF_census_logger.total_time}==========={len(ADF_census_id)}===========")
df.to_csv(file_path, index=False)


# EIDIG
print('EIDIG begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['EIDIG'], 'number': ['-'], 'time': ['-'], 'global_time': ['-'],
                             'local_time': ['-']}, index=[0]), ignore_index=True)

EIDIG_credit_id, EIDIG_credit_logger = EIDIG.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'number': [len(EIDIG_credit_id)], 'time': [EIDIG_credit_logger.total_time],
                             'global_time': [EIDIG_credit_logger.global_time], 'local_time': [EIDIG_credit_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={EIDIG_credit_logger.total_time}==========={len(EIDIG_credit_id)}===========")

EIDIG_bank_id, EIDIG_bank_logger = EIDIG.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'number': [len(EIDIG_bank_id)], 'time': [EIDIG_bank_logger.total_time],
                             'global_time': [EIDIG_bank_logger.global_time], 'local_time': [EIDIG_bank_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={EIDIG_bank_logger.total_time}==========={len(EIDIG_bank_id)}===========")

EIDIG_census_id, EIDIG_census_logger = EIDIG.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'number': [len(EIDIG_census_id)], 'time': [EIDIG_census_logger.total_time],
                             'global_time': [EIDIG_census_logger.global_time], 'local_time': [EIDIG_census_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={EIDIG_census_logger.total_time}==========={len(EIDIG_census_id)}===========")
df.to_csv(file_path, index=False)


# NeuronFair
print('NeuronFair begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['NeuronFair'], 'number': ['-'], 'time': ['-'], 'global_time': ['-'],
                             'local_time': ['-']}, index=[0]), ignore_index=True)

NeuronFair_credit_id, NeuronFair_credit_logger = NeuronFair.individual_discrimination_generation('credit', config.Credit
                                                                                                 , credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'number': [len(NeuronFair_credit_id)], 'time': [NeuronFair_credit_logger.total_time],
                             'global_time': [NeuronFair_credit_logger.global_time], 'local_time': [NeuronFair_credit_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={NeuronFair_credit_logger.total_time}==========={len(NeuronFair_credit_id)}===========")

NeuronFair_bank_id, NeuronFair_bank_logger = NeuronFair.individual_discrimination_generation('bank', config.Bank,
                                                                                             bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'number': [len(NeuronFair_bank_id)], 'time': [NeuronFair_bank_logger.total_time],
                             'global_time': [NeuronFair_bank_logger.global_time], 'local_time': [NeuronFair_bank_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={NeuronFair_bank_logger.total_time}==========={len(NeuronFair_bank_id)}===========")

NeuronFair_census_id, NeuronFair_census_logger = NeuronFair.individual_discrimination_generation('census', config.Census
                                                                                                 , census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'number': [len(NeuronFair_census_id)], 'time': [NeuronFair_census_logger.total_time],
                             'global_time': [NeuronFair_census_logger.global_time], 'local_time': [NeuronFair_census_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={NeuronFair_census_logger.total_time}==========={len(NeuronFair_census_id)}===========")
df.to_csv(file_path, index=False)


# FEDIG
print('FEDIG begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['FEDIG'], 'number': ['-'], 'time': ['-'], 'global_time': ['-'],
                             'local_time': ['-']}, index=[0]), ignore_index=True)

FEDIG_credit_id, FEDIG_credit_logger = FEDIG.individual_discrimination_generation('credit', config.Credit, credit_model,
                                                                                  decay=0.2, c_num=4, min_len=250,
                                                                                  delta1=0.1, delta2=0.2)
df = df.append(pd.DataFrame({'data': ['credit'], 'number': [len(FEDIG_credit_id)], 'time': [FEDIG_credit_logger.total_time],
                             'global_time': [FEDIG_credit_logger.global_time], 'local_time': [FEDIG_credit_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={FEDIG_credit_logger.total_time}==========={len(FEDIG_credit_id)}===========")

FEDIG_bank_id, FEDIG_bank_logger = FEDIG.individual_discrimination_generation('bank', config.Bank, bank_model, decay=0.2
                                                                              , c_num=4, min_len=250, delta1=0.15,
                                                                              delta2=0.35)
df = df.append(pd.DataFrame({'data': ['bank'], 'number': [len(FEDIG_bank_id)], 'time': [FEDIG_bank_logger.total_time],
                             'global_time': [FEDIG_bank_logger.global_time], 'local_time': [FEDIG_bank_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={FEDIG_bank_logger.total_time}==========={len(FEDIG_bank_id)}===========")

FEDIG_census_id, FEDIG_census_logger = FEDIG.individual_discrimination_generation('census', config.Census, census_model,
                                                                                  decay=0.2, c_num=4, min_len=250,
                                                                                  delta1=0.1, delta2=0.3)
df = df.append(pd.DataFrame({'data': ['census'], 'number': [len(FEDIG_census_id)], 'time': [FEDIG_census_logger.total_time],
                             'global_time': [FEDIG_census_logger.global_time], 'local_time': [FEDIG_census_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={FEDIG_census_logger.total_time}==========={len(FEDIG_census_id)}===========")
df.to_csv(file_path, index=False)


# DICE
print('DICE begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['DICE'], 'number': ['-'], 'time': ['-'], 'global_time': ['-'],
                             'local_time': ['-']}, index=[0]), ignore_index=True)
DICE_credit_id, DICE_credit_logger = DICE.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'number': [len(DICE_credit_id)], 'time': [DICE_credit_logger.total_time],
                             'global_time': [DICE_credit_logger.global_time], 'local_time': [DICE_credit_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={DICE_credit_logger.total_time}==========={len(DICE_credit_id)}===========")

DICE_bank_id, DICE_bank_logger = DICE.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'number': [len(DICE_bank_id)], 'time': [DICE_bank_logger.total_time],
                             'global_time': [DICE_bank_logger.global_time], 'local_time': [DICE_bank_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={DICE_bank_logger.total_time}==========={len(DICE_bank_id)}===========")

DICE_census_id, DICE_census_logger = DICE.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'number': [len(DICE_census_id)], 'time': [DICE_census_logger.total_time],
                             'global_time': [DICE_census_logger.global_time], 'local_time': [DICE_census_logger.local_time]}, index=[0]), ignore_index=True)
print(f"==========={DICE_census_logger.total_time}==========={len(DICE_census_id)}===========")

df = df.append(pd.DataFrame({'data': ['-'], 'number': ['-'], 'time': ['-'], 'global_time': ['-'],
                             'local_time': ['-']}, index=[0]), ignore_index=True)
df.to_csv(file_path, index=False)
# experiment ends
