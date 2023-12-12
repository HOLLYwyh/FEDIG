"""
 This file evaluates the discriminatory degree of retrained models.
"""

import sys
import numpy as np
from tensorflow.keras.models import load_model
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
    statistics = np.empty(shape=(0, ))
    for i in range(sample_round):
        gen_id = random_generate(num_attrs, protected_features, constraint, model, generation_num)
        percentage = len(gen_id) / generation_num
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)


def evaluate_discrimination(sample_round, generation_num, info, model):
    num_attrs = len(info.X[0])
    idi_percentage(sample_round, generation_num, num_attrs, info.protected_attrs, info.constraint, model)


# load models
# origin models
credit_origin_path = '../models/trained_models/credit_model.h5'
bank_origin_path = '../models/trained_models/bank_model.h5'
census_origin_path = '../../models/trained_models/census_model.h5'

# retrained models
credit_5_retrained_path = '../models/retrained_models/credit_final_retrained_model.h5'
bank_retrained_path = '../models/retrained_models/bank_final_retrained_model.h5'

credit_origin_model = load_model(credit_origin_path)
bank_origin_model = load_model(bank_origin_path)

credit_retrained_model = load_model(credit_5_retrained_path)
bank_retrained_model = load_model(bank_retrained_path)

# evaluate retrained models
evaluate_discrimination(10, 100, config.Credit, credit_origin_model)

evaluate_discrimination(10, 100,  config.Credit, credit_retrained_model)

evaluate_discrimination(10, 100, config.Bank, bank_origin_model)

evaluate_discrimination(10, 100,  config.Bank, bank_retrained_model)







