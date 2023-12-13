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

# evaluate retrained models
# credit
evaluate_discrimination(100, 100, config.Credit, credit_origin_model)
evaluate_discrimination(100, 100, config.Credit, credit_5_retrained_model)
evaluate_discrimination(100, 100, config.Credit, credit_10_retrained_model)
evaluate_discrimination(100, 100, config.Credit, credit_15_retrained_model)
evaluate_discrimination(100, 100, config.Credit, credit_20_retrained_model)
evaluate_discrimination(100, 100, config.Credit, credit_25_retrained_model)
evaluate_discrimination(100, 100, config.Credit, credit_30_retrained_model)

# bank
# evaluate_discrimination(100, 100, config.Bank, bank_origin_model)
# evaluate_discrimination(100, 100, config.Bank, bank_5_retrained_model)
# evaluate_discrimination(100, 100, config.Bank, bank_10_retrained_model)
# evaluate_discrimination(100, 100, config.Bank, bank_15_retrained_model)
# evaluate_discrimination(100, 100, config.Bank, bank_20_retrained_model)
# evaluate_discrimination(100, 100, config.Bank, bank_25_retrained_model)
# evaluate_discrimination(100, 100, config.Bank, bank_30_retrained_model)

# census
# evaluate_discrimination(100, 100, config.Census, census_origin_model)
# evaluate_discrimination(100, 100, config.Census, census_5_retrained_model)
# evaluate_discrimination(100, 100, config.Census, census_10_retrained_model)
# evaluate_discrimination(100, 100, config.Census, census_15_retrained_model)
# evaluate_discrimination(100, 100, config.Census, census_20_retrained_model)
# evaluate_discrimination(100, 100, config.Census, census_25_retrained_model)
# evaluate_discrimination(100, 100, config.Census, census_30_retrained_model)


"""
    .95 confidence
1. credit:
    - original: 0.1948 ± 0.007214717136520323
    - 5%:       0.10309999999999998 ± 0.005678556703952158
    - 10%:      0.10469999999999999 ± 0.005477174322586419
    - 15%:      0.0787 ± 0.005278589675282593
    - 20%:      0.1506 ± 0.005988882553532003
    - 25%:      0.1029 ± 0.005187857499970485
    - 30%:      0.08839999999999999 ± 0.005123966338687247
2. bank:
    - original: 
    - 5%:
    - 10%:
    - 15%:
    - 20%:
    - 25%:
    - 30%:
3. census:
    - original: 
    - 5%:
    - 10%:
    - 15%:
    - 20%:
    - 25%:
    - 30%:
"""


