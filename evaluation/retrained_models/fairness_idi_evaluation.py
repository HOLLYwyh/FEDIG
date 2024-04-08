"""
 This file evaluates the discriminatory degree of IDI retrained models.
"""

import sys
import numpy as np
from keras.models import load_model

sys.path.append('../')
from utils import config
from utils import utils


def random_generate(num_attrs, protected_features, constraint, model, generation_num):
    gen_id = np.empty(shape=(0, num_attrs))
    for i in range(generation_num):
        x_picked = [0] * num_attrs
        for a in range(num_attrs):
            x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1] + 1)
        if utils.is_discriminatory(x_picked, utils.get_similar_set(x_picked, num_attrs, protected_features, constraint),
                                   model):
            gen_id = np.append(gen_id, [x_picked], axis=0)
    return gen_id


def idi_percentage(sample_round, generation_num, num_attrs, protected_features, constraint, model):
    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        gen_id = random_generate(num_attrs, protected_features, constraint, model, generation_num)
        percentage = len(gen_id) / generation_num
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)


def evaluate_discrimination(sample_round, generation_num, info, num_attrs, model):
    idi_percentage(sample_round, generation_num, num_attrs, info.protected_attrs, info.constraint, model)


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
credit_foreign_idi_path = '../models/idi_trained_models/credit_foreign_idi_model.h5'
credit_house_idi_path = '../models/idi_trained_models/credit_house_idi_model.h5'
credit_skilled_idi_path = '../models/idi_trained_models/credit_skilled_idi_model.h5'
credit_unskilled_idi_path = '../models/idi_trained_models/credit_unskilled_idi_model.h5'
# bank
bank_day_idi_path = '../models/idi_trained_models/bank_day_idi_model.h5'
bank_marital_idi_path = '../models/idi_trained_models/bank_marital_idi_model.h5'
bank_month_idi_path = '../models/idi_trained_models/bank_month_idi_model.h5'
# census
census_occupation_idi_path = '../models/idi_trained_models/census_occupation_idi_model.h5'
census_work_class_idi_path = '../models/idi_trained_models/census_workclass_idi_model.h5'

credit_foreign_idi_model = load_model(credit_foreign_idi_path)
credit_house_idi_model = load_model(credit_house_idi_path)
credit_skilled_idi_model = load_model(credit_skilled_idi_path)
credit_unskilled_idi_model = load_model(credit_unskilled_idi_path)

bank_day_idi_model = load_model(bank_day_idi_path)
bank_marital_idi_model = load_model(bank_marital_idi_path)
bank_month_idi_model = load_model(bank_month_idi_path)

census_occupation_idi_model = load_model(census_occupation_idi_path)
census_work_class_idi_model = load_model(census_work_class_idi_path)

round_num = 100
generate_num = 100
credit_num_attrs = len(config.Credit.X[0])
credit_idi_num_attrs = len(config.Credit.X[0]) - 1
bank_num_attrs = len(config.Bank.X[0])
bank_idi_num_attrs = len(config.Bank.X[0]) - 1
census_num_attrs = len(config.Census.X[0])
census_idi_num_attrs = len(config.Census.X[0]) - 1

# evaluate retrained models
# credit
evaluate_discrimination(round_num, generate_num, config.Credit, credit_num_attrs, credit_origin_model)
evaluate_discrimination(round_num, generate_num, config.Credit, credit_idi_num_attrs, credit_foreign_idi_model)
evaluate_discrimination(round_num, generate_num, config.Credit, credit_idi_num_attrs, credit_house_idi_model)
evaluate_discrimination(round_num, generate_num, config.Credit, credit_idi_num_attrs, credit_skilled_idi_model)
evaluate_discrimination(round_num, generate_num, config.Credit, credit_idi_num_attrs, credit_unskilled_idi_model)

# bank
evaluate_discrimination(round_num, generate_num, config.Bank, bank_num_attrs, bank_origin_model)
evaluate_discrimination(round_num, generate_num, config.Bank, bank_idi_num_attrs, bank_day_idi_model)
evaluate_discrimination(round_num, generate_num, config.Bank, bank_idi_num_attrs, bank_marital_idi_model)
evaluate_discrimination(round_num, generate_num, config.Bank, bank_idi_num_attrs, bank_month_idi_model)

# census
evaluate_discrimination(round_num, generate_num, config.Census, census_num_attrs, census_origin_model)
evaluate_discrimination(round_num, generate_num, config.Census, census_idi_num_attrs, census_occupation_idi_model)
evaluate_discrimination(round_num, generate_num, config.Census, census_idi_num_attrs, census_work_class_idi_model)

"""
    .95 confidence
1. credit:
    - original:     0.1892 ± 0.007352900227801272
    - foreign:      0.10300000000000001 ± 0.00696558741241541
    - house:        0.11979999999999999 ± 0.005925427187975564
    - skilled:      0.053399999999999996 ± 0.004393373992730416
    - unskilled:    0.08829999999999999 ± 0.005530222577799197
 
2. bank:
    - original:     0.10829999999999998 ± 0.006223146612446152
    - day:          0.05030000000000001 ± 0.00450761650542723
    - marital:      0.030100000000000002 ± 0.0034675385852215115
    - month:        0.05499999999999999 ± 0.0051893752995905

3. census:
    - original:     0.4418999999999999 ± 0.0089290639061438
    - occupation:   0.29500000000000004 ± 0.009340382005035982
    - work_class:   0.2289 ± 0.007819859630453733

"""
