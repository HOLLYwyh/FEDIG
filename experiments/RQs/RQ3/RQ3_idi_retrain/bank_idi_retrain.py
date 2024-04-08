"""
This file is the experiments of RQ3:
    - IDI training on bank dataset
    - Irrelevant features : 'month', 'marital', 'day'
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
data_path = '../datasets/bank'
df = pd.read_csv(data_path)

# raw data
raw_data_path = '../datasets/raw/bank-full.csv'
raw_df = pd.read_csv(raw_data_path, sep=';')

month_id = raw_df.columns.tolist().index('month')
marital_id = raw_df.columns.tolist().index('marital')
day_id = raw_df.columns.tolist().index('day')

# load IDIs
idi = np.load('./logfile/generated_instances/bank_discriminatory_instance.npy')
classifier = joblib.load('../models/ensemble_models/bank_ensemble.pkl')

protected_features = config.Bank.protected_attrs
selected_indices = np.random.choice(idi.shape[0], size=9042, replace=False)
idi_selected = idi[selected_indices, :]
vote_label = classifier.predict(np.delete(idi_selected, protected_features, axis=1))

# create and train a six-layer neural network with bank dataset
model_name = ['month', 'marital', 'day']
ids = [month_id, marital_id, day_id]

for name, i in zip(model_name, ids):
    data = df.drop(df.columns[i], axis=1, inplace=False).values
    name_idi = np.hstack((idi_selected[:, :i], idi_selected[:, i + 1:]))

    x = data[:, :-1]
    y = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=821)
    x_train = np.append(x_train, name_idi, axis=0)
    y_train = np.append(y_train, vote_label, axis=0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=821)

    # create and train a six-layer neural network with bank dataset
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=x_train.shape[1:]),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))
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
    model.save('../models/idi_trained_models/bank_' + name + '_idi_model.h5')
    print(name + " bank model has been saved...")


"""
month model:
    test_loss:          0.25672492384910583
    test_accuracy:      0.8993696570396423
    Precision rate:     0.6970802919708029
    Recall rate:        0.18762278978389

marital model:
    test_loss:          0.2574494481086731
    test_accuracy:      0.9009178280830383
    Precision rate:     0.7147887323943662
    Recall rate:        0.1994106090373281
    
day model:
    test_loss:          0.2577890455722809
    test_accuracy:      0.9001437425613403
    Precision rate:     0.7046263345195729
    Recall rate:        0.1944990176817289
"""

