"""
This file will divide the datasets we preprocessed before into clusters by using K-Means algorithm.
"""

import joblib
import pandas as pd
from sklearn.cluster import KMeans

print('Start clustering by using K-Means')

# load dataset
bank_data_path = '../datasets/bank'
census_data_path = '../datasets/census'
credit_data_path = '../datasets/credit'

bank_df = pd.read_csv(bank_data_path)
census_df = pd.read_csv(census_data_path)
credit_df = pd.read_csv(credit_data_path)

bank_data = bank_df.values
census_data = census_df.values
credit_data = credit_df.values


# kmeans
c_num = 4
bank_kmeans = KMeans(n_clusters=c_num).fit(bank_data)
census_kmeans = KMeans(n_clusters=c_num).fit(census_data)
credit_kmeans = KMeans(n_clusters=c_num).fit(credit_data)

bank_cluster = {'data': bank_data, 'cluster_labels': bank_kmeans.labels_}
census_cluster = {'data': census_data, 'cluster_labels': census_kmeans.labels_}
credit_cluster = {'data': credit_data, 'cluster_labels': credit_kmeans.labels_}

# save the clusters
joblib.dump(bank_cluster, 'bank.pkl')
joblib.dump(census_cluster, 'census.pkl')
joblib.dump(credit_cluster, 'credit.pkl')

print('Finished')





