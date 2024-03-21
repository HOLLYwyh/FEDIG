"""
This file is the experiments of RQ1:
    - Effectiveness of FEDIG
"""

import sys
import pandas as pd
import ADF
import EIDIG
import NeuronFair
import DICE
import FEDIG

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
file_path = 'logfile/RQs/RQ1.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['data', 'all_num', 'all_nd_num', 'g_num', 'g_nd_num', 'l_num', 'l_nd_num', 'time'])
    df.to_csv(file_path, index=False)

# experiment begins
# ADF
print('ADF begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['ADF'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
ADF_credit_logger = ADF.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'all_num': [ADF_credit_logger.all_number], 'all_nd_num': [ADF_credit_logger.all_non_duplicate_number],
                             'g_num': [ADF_credit_logger.global_number], 'g_nd_num': [ADF_credit_logger.global_non_duplicate_number],
                             'l_num': [ADF_credit_logger.local_number], 'l_nd_num': [ADF_credit_logger.local_non_duplicate_number], 'time': [ADF_credit_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={ADF_credit_logger.all_non_duplicate_number}==========={ADF_credit_logger.all_number}"
      f"==========={ADF_credit_logger.total_time}===========")


ADF_bank_logger = ADF.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'all_num': [ADF_bank_logger.all_number], 'all_nd_num': [ADF_bank_logger.all_non_duplicate_number],
                             'g_num': [ADF_bank_logger.global_number], 'g_nd_num': [ADF_bank_logger.global_non_duplicate_number],
                             'l_num': [ADF_bank_logger.local_number], 'l_nd_num': [ADF_bank_logger.local_non_duplicate_number], 'time': [ADF_bank_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={ADF_bank_logger.all_non_duplicate_number}==========={ADF_bank_logger.all_number}"
      f"==========={ADF_bank_logger.total_time}===========")

ADF_census_logger = ADF.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'all_num': [ADF_census_logger.all_number], 'all_nd_num': [ADF_census_logger.all_non_duplicate_number],
                             'g_num': [ADF_census_logger.global_number], 'g_nd_num': [ADF_census_logger.global_non_duplicate_number],
                             'l_num': [ADF_census_logger.local_number], 'l_nd_num': [ADF_census_logger.local_non_duplicate_number], 'time': [ADF_census_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={ADF_census_logger.all_non_duplicate_number}==========={ADF_census_logger.all_number}"
      f"==========={ADF_census_logger.total_time}===========")
df.to_csv(file_path, index=False)


# EIDIG
print('EIDIG begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['EIDIG'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time':['-']},
                            index=[0]), ignore_index=True)
EIDIG_credit_logger = EIDIG.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'all_num': [EIDIG_credit_logger.all_number], 'all_nd_num': [EIDIG_credit_logger.all_non_duplicate_number],
                             'g_num': [EIDIG_credit_logger.global_number], 'g_nd_num': [EIDIG_credit_logger.global_non_duplicate_number],
                             'l_num': [EIDIG_credit_logger.local_number], 'l_nd_num': [EIDIG_credit_logger.local_non_duplicate_number], 'time': [EIDIG_credit_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={EIDIG_credit_logger.all_non_duplicate_number}==========={EIDIG_credit_logger.all_number}"
      f"==========={EIDIG_credit_logger.total_time}===========")


EIDIG_bank_logger = EIDIG.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'all_num': [EIDIG_bank_logger.all_number], 'all_nd_num': [EIDIG_bank_logger.all_non_duplicate_number],
                             'g_num': [EIDIG_bank_logger.global_number], 'g_nd_num': [EIDIG_bank_logger.global_non_duplicate_number],
                             'l_num': [EIDIG_bank_logger.local_number], 'l_nd_num': [EIDIG_bank_logger.local_non_duplicate_number], 'time': [EIDIG_bank_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={EIDIG_bank_logger.all_non_duplicate_number}==========={EIDIG_bank_logger.all_number}"
      f"==========={EIDIG_bank_logger.total_time}===========")

EIDIG_census_logger = EIDIG.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'all_num': [EIDIG_census_logger.all_number], 'all_nd_num': [EIDIG_census_logger.all_non_duplicate_number],
                             'g_num': [EIDIG_census_logger.global_number], 'g_nd_num': [EIDIG_census_logger.global_non_duplicate_number],
                             'l_num': [EIDIG_census_logger.local_number], 'l_nd_num': [EIDIG_census_logger.local_non_duplicate_number], 'time': [EIDIG_census_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={EIDIG_census_logger.all_non_duplicate_number}==========={EIDIG_census_logger.all_number}"
      f"==========={EIDIG_census_logger.total_time}===========")
df.to_csv(file_path, index=False)


# NeuronFair
print('NeuronFair begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['NeuronFair'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
NeuronFair_credit_logger = NeuronFair.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'all_num': [NeuronFair_credit_logger.all_number], 'all_nd_num': [NeuronFair_credit_logger.all_non_duplicate_number],
                             'g_num': [NeuronFair_credit_logger.global_number], 'g_nd_num': [NeuronFair_credit_logger.global_non_duplicate_number],
                             'l_num': [NeuronFair_credit_logger.local_number], 'l_nd_num': [NeuronFair_credit_logger.local_non_duplicate_number], 'time': [NeuronFair_credit_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={NeuronFair_credit_logger.all_non_duplicate_number}==========={NeuronFair_credit_logger.all_number}"
      f"==========={NeuronFair_credit_logger.total_time}===========")


NeuronFair_bank_logger = NeuronFair.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'all_num': [NeuronFair_bank_logger.all_number], 'all_nd_num': [NeuronFair_bank_logger.all_non_duplicate_number],
                             'g_num': [NeuronFair_bank_logger.global_number], 'g_nd_num': [NeuronFair_bank_logger.global_non_duplicate_number],
                             'l_num': [NeuronFair_bank_logger.local_number], 'l_nd_num': [NeuronFair_bank_logger.local_non_duplicate_number], 'time': [NeuronFair_bank_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={NeuronFair_bank_logger.all_non_duplicate_number}==========={NeuronFair_bank_logger.all_number}"
      f"==========={NeuronFair_bank_logger.total_time}===========")

NeuronFair_census_logger = NeuronFair.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'all_num': [NeuronFair_census_logger.all_number], 'all_nd_num': [NeuronFair_census_logger.all_non_duplicate_number],
                             'g_num': [NeuronFair_census_logger.global_number], 'g_nd_num': [NeuronFair_census_logger.global_non_duplicate_number],
                             'l_num': [NeuronFair_census_logger.local_number], 'l_nd_num': [NeuronFair_census_logger.local_non_duplicate_number], 'time': [NeuronFair_census_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={NeuronFair_census_logger.all_non_duplicate_number}==========={NeuronFair_census_logger.all_number}"
      f"==========={NeuronFair_census_logger.total_time}===========")
df.to_csv(file_path, index=False)


# FEDIG
print('FEDIG begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['FEDIG'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
FEDIG_credit_logger = FEDIG.individual_discrimination_generation('credit', config.Credit, credit_model, decay=0.2,
                                                               min_len=250)
df = df.append(pd.DataFrame({'data': ['credit'], 'all_num': [FEDIG_credit_logger.all_number], 'all_nd_num': [FEDIG_credit_logger.all_non_duplicate_number],
                             'g_num': [FEDIG_credit_logger.global_number], 'g_nd_num': [FEDIG_credit_logger.global_non_duplicate_number],
                             'l_num': [FEDIG_credit_logger.local_number], 'l_nd_num': [FEDIG_credit_logger.local_non_duplicate_number], 'time': [FEDIG_credit_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={FEDIG_credit_logger.all_non_duplicate_number}==========={FEDIG_credit_logger.all_number}"
      f"==========={FEDIG_credit_logger.total_time}===========")


FEDIG_bank_logger = FEDIG.individual_discrimination_generation('bank', config.Bank, bank_model, decay=0.2,
                                                               min_len=250)
df = df.append(pd.DataFrame({'data': ['bank'], 'all_num': [FEDIG_bank_logger.all_number], 'all_nd_num': [FEDIG_bank_logger.all_non_duplicate_number],
                             'g_num': [FEDIG_bank_logger.global_number], 'g_nd_num': [FEDIG_bank_logger.global_non_duplicate_number],
                             'l_num': [FEDIG_bank_logger.local_number], 'l_nd_num': [FEDIG_bank_logger.local_non_duplicate_number], 'time': [FEDIG_bank_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={FEDIG_bank_logger.all_non_duplicate_number}==========={FEDIG_bank_logger.all_number}"
      f"==========={FEDIG_bank_logger.total_time}===========")

FEDIG_census_logger = FEDIG.individual_discrimination_generation('census', config.Census, census_model, decay=0.2,
                                                               min_len=250)
df = df.append(pd.DataFrame({'data': ['census'], 'all_num': [FEDIG_census_logger.all_number], 'all_nd_num': [FEDIG_census_logger.all_non_duplicate_number],
                             'g_num': [FEDIG_census_logger.global_number], 'g_nd_num': [FEDIG_census_logger.global_non_duplicate_number],
                             'l_num': [FEDIG_census_logger.local_number], 'l_nd_num': [FEDIG_census_logger.local_non_duplicate_number], 'time': [FEDIG_census_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={FEDIG_census_logger.all_non_duplicate_number}==========={FEDIG_census_logger.all_number}"
      f"==========={FEDIG_census_logger.total_time}===========")
df.to_csv(file_path, index=False)

# DICE
print('DICE begins...')
df = pd.read_csv(file_path)
df = df.append(pd.DataFrame({'data': ['DICE'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
DICE_credit_logger = DICE.individual_discrimination_generation('credit', config.Credit, credit_model)
df = df.append(pd.DataFrame({'data': ['credit'], 'all_num': [DICE_credit_logger.all_number], 'all_nd_num': [DICE_credit_logger.all_non_duplicate_number],
                             'g_num': [DICE_credit_logger.global_number], 'g_nd_num': [DICE_credit_logger.global_non_duplicate_number],
                             'l_num': [DICE_credit_logger.local_number], 'l_nd_num': [DICE_credit_logger.local_non_duplicate_number], 'time': [DICE_credit_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={DICE_credit_logger.all_non_duplicate_number}==========={DICE_credit_logger.all_number}"
      f"==========={DICE_credit_logger.total_time}===========")


DICE_bank_logger = DICE.individual_discrimination_generation('bank', config.Bank, bank_model)
df = df.append(pd.DataFrame({'data': ['bank'], 'all_num': [DICE_bank_logger.all_number], 'all_nd_num': [DICE_bank_logger.all_non_duplicate_number],
                             'g_num': [DICE_bank_logger.global_number], 'g_nd_num': [DICE_bank_logger.global_non_duplicate_number],
                             'l_num': [DICE_bank_logger.local_number], 'l_nd_num': [DICE_bank_logger.local_non_duplicate_number], 'time': [DICE_bank_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={DICE_bank_logger.all_non_duplicate_number}==========={DICE_bank_logger.all_number}"
      f"==========={DICE_bank_logger.total_time}===========")

DICE_census_logger = DICE.individual_discrimination_generation('census', config.Census, census_model)
df = df.append(pd.DataFrame({'data': ['census'], 'all_num': [DICE_census_logger.all_number], 'all_nd_num': [DICE_census_logger.all_non_duplicate_number],
                             'g_num': [DICE_census_logger.global_number], 'g_nd_num': [DICE_census_logger.global_non_duplicate_number],
                             'l_num': [DICE_census_logger.local_number], 'l_nd_num': [DICE_census_logger.local_non_duplicate_number], 'time': [DICE_census_logger.total_time]},
                            index=[0]), ignore_index=True)
print(f"==========={DICE_census_logger.all_non_duplicate_number}==========={DICE_census_logger.all_number}"
      f"==========={DICE_census_logger.total_time}===========")
df.to_csv(file_path, index=False)


df = df.append(pd.DataFrame({'data': ['-'], 'all_num': ['-'], 'all_nd_num': ['-'],
                             'g_num': ['-'], 'g_nd_num': ['-'],
                             'l_num': ['-'], 'l_nd_num': ['-'], 'time': ['-']},
                            index=[0]), ignore_index=True)
df.to_csv(file_path, index=False)
# experiment ends
