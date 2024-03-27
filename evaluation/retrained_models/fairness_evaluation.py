"""
 This file evaluates the discriminatory degree of retrained models.
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
evaluate_discrimination(100, 100, config.Bank, bank_origin_model)
evaluate_discrimination(100, 100, config.Bank, bank_5_retrained_model)
evaluate_discrimination(100, 100, config.Bank, bank_10_retrained_model)
evaluate_discrimination(100, 100, config.Bank, bank_15_retrained_model)
evaluate_discrimination(100, 100, config.Bank, bank_20_retrained_model)
evaluate_discrimination(100, 100, config.Bank, bank_25_retrained_model)
evaluate_discrimination(100, 100, config.Bank, bank_30_retrained_model)

# census
evaluate_discrimination(100, 100, config.Census, census_origin_model)
evaluate_discrimination(100, 100, config.Census, census_5_retrained_model)
evaluate_discrimination(100, 100, config.Census, census_10_retrained_model)
evaluate_discrimination(100, 100, config.Census, census_15_retrained_model)
evaluate_discrimination(100, 100, config.Census, census_20_retrained_model)
evaluate_discrimination(100, 100, config.Census, census_25_retrained_model)
evaluate_discrimination(100, 100, config.Census, census_30_retrained_model)


"""
    .95 confidence
1. credit:
    - original: 0.1925 ± 0.008344974775276437
    - 5%:       0.0983 ± 0.005788175684963269
    - 10%:      0.10649999999999998 ± 0.006106744795715636
    - 15%:      0.0754 ± 0.005030904236814691
    - 20%:      0.1462 ± 0.006720142331826015
    - 25%:      0.10069999999999998 ± 0.006017263843309515
    - 30%:      0.0883 ± 0.006129851038973133
2. bank:
    - original: 0.1146 ± 0.006746614961593703
    - 5%:       0.0438 ± 0.003828577929205569
    - 10%:      0.0447 ± 0.004096868872688018
    - 15%:      0.0424 ± 0.003871880659317898
    - 20%:      0.0385 ± 0.003443971544597894
    - 25%:      0.0382 ± 0.0035321353541448554
    - 30%:      0.0326 ± 0.004022729898961649
3. census:
    - original: 0.4584 ± 0.010819910121623009
    - 5%:       0.42749999999999994 ± 0.009319279585890747
    - 10%:      0.336 ± 0.008421160015104807
    - 15%:      0.31789999999999996 ± 0.009789390657237046
    - 20%:      0.2711 ± 0.009100375192265427
    - 25%:      0.2631 ± 0.008084590171431079
    - 30%:      0.29200000000000004 ± 0.008659563037474814
"""


