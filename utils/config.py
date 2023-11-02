"""
Some configs and constraints of datasets
"""

import numpy as np
import pandas as pd

bank_data_path = '../datasets/bank'
bank_df = pd.read_csv(bank_data_path)
bank_data = bank_df.values

census_data_path = '../datasets/census'
census_df = pd.read_csv(census_data_path)
census_data = census_df.values

credit_data_path = '../datasets/credit'
credit_df = pd.read_csv(credit_data_path)
credit_data = credit_df.values


class Bank:
    X = bank_data[:, :-1]

    # constraints
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

    # age(0) is the protected attribute
    protected_attrs = [0]


class Census:
    X = census_data[:, :-1]

    # constraints
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

    # age(0), race(7), gender(8) are protected attributes
    protected_attrs = [0, 7, 8]


class Credit:
    X = credit_data[:, 1:]

    # constraints
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

    # gender(6), age(9) are protected attributes
    protected_attrs = [6, 9]

