"""
- This file provides the essential functions for individual discrimination generation.
- For DICE
"""

import random
import numpy as np


# select the seed inputs for fairness testing
def seed_test_input(clusters, limit):
    rows = []
    max_size = max([len(c) for c in clusters])
    for i in range(max_size):
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c):
                continue
            rows.append(c[i])
            if len(rows) == limit:
                break
    return np.array(rows)


# generate all the predict value of x_similar_set
def generate_predict(similar_x_set, model):
    predict = model(similar_x_set)
    return predict.numpy().reshape(1, -1)


# cluster the similar_set with epsilon
def clustering(predict, similar_x_set, epsilon):
    cluster_dic = {}
    bins = np.arange(0, 1, epsilon)
    digitized = (np.digitize(predict, bins) - 1)[0]

    for i in range(len(similar_x_set)):
        bin_index = digitized[i]
        if bin_index not in cluster_dic:
            cluster_dic[bin_index] = [similar_x_set[i]]
        else:
            cluster_dic[bin_index].append(similar_x_set[i])
    return cluster_dic


# random select two samples with the cluster_dictionary
def global_sample_select(cluster_dic):
    max_len = 0
    max_index = 0
    for i in cluster_dic.keys():
        if len(cluster_dic[i]) > max_len:
            max_len = len(cluster_dic[i])
            max_index = i

    if max_len > 1:
        x1, x2 = random.sample(cluster_dic[max_index], 2)
    else:
        index1, index2 = random.sample(cluster_dic.keys(), 2)
        x1 = cluster_dic[index1][0]
        x2 = cluster_dic[index2][0]
    return x1, x2

