"""
This file is the experiments of RQ3:
    - IDI training on credit dataset
    - Irrelevant features : 'job_skilledVsMgt', 'jobs_unskilledVsMgt', 'house_ownsVsFree', 'ForeignWorker'
"""
import sys
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

sys.path.append('..')
from utils import config

# load data preprocessed before
data_path = '../datasets/credit'
df = pd.read_csv(data_path)

# raw data
raw_data_path = '../datasets/raw/proc_german_num_02 withheader-2.csv'
raw_df = pd.read_csv(raw_data_path)

skilled_id = raw_df.columns.tolist().index('job_skilledVsMgt')
unskilled_id = raw_df.columns.tolist().index('jobs_unskilledVsMgt')
house_id = raw_df.columns.tolist().index('house_ownsVsFree')
foreign_id = raw_df.columns.tolist().index('ForeignWorker')

# load IDIs
idi = np.load('./logfile/generated_instances/credit_discriminatory_instance.npy')
classifier = joblib.load('../models/ensemble_models/credit_ensemble.pkl')

protected_features = config.Credit.protected_attrs
selected_indices = np.random.choice(idi.shape[0], size=150, replace=False)
idi_selected = idi[selected_indices, :]
vote_label = classifier.predict(np.delete(idi_selected, protected_features, axis=1))

# create and train a six-layer neural network with credit dataset
model_name = ['skilled', 'unskilled', 'house', 'foreign']
ids = [skilled_id, unskilled_id, house_id, foreign_id]

for name, i in zip(model_name, ids):
    data = df.drop(df.columns[i], axis=1, inplace=False).values
    name_idi = np.hstack((idi_selected[:, :i-1], idi_selected[:, i:]))

    x = data[:, 1:]
    y = data[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1001)
    x_train = np.append(x_train, name_idi, axis=0)
    y_train = np.append(y_train, vote_label, axis=0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1001)

    # create and train a six-layer neural network with credit dataset
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=x_train.shape[1:]),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=25, validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_test, y_test)

    # calculate the precision rate and recall rate
    y_predict = model.predict(x_test)
    y_predict_binary = (y_predict > 0.5).astype(int)
    precision = precision_score(y_test, y_predict_binary)
    recall = recall_score(y_test, y_predict_binary)

    print(f"test_loss:{test_loss}")
    print(f"test_accuracy:{test_acc}")
    print(f"Precision rate: {precision}")
    print(f"Recall rate:{recall}")

    # save the model
    model.save('../models/idi_trained_models/credit_' + name + '_idi_model.h5')
    print(name + " credit model has been saved...")

"""
skilled model:
    test_loss:          0.46529659628868103
    test_accuracy:      0.75
    Precision rate:     0.5102040816326531
    Recall rate:        0.49019607843137253

unskilled model:
    test_loss:          0.47474756836891174
    test_accuracy:      0.7699999809265137
    Precision rate:     0.5609756097560976
    Recall rate:        0.45098039215686275

house model:
    test_loss:          0.49555161595344543
    test_accuracy:      0.7649999856948853
    Precision rate:     0.5416666666666666
    Recall rate:        0.5098039215686274

foreign model:
    test_loss:          0.5734724998474121
    test_accuracy:      0.7350000143051147
    Precision rate:     0.6216216216216216
    Recall rate:        0.3709677419354839
"""

