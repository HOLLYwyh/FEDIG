"""
- This file constructs training set, test set.
- This file also trains the DNN model of Bank Marketing Dataset
"""
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# load the data we preprocessed before
data_path = '../datasets/bank'
df = pd.read_csv(data_path)
data = df.values

# spilt the data into training dataset, validation dataset and test dataset
X = data[:, :-1]
Y = data[:, -1]
X_train_all, X_test, Y_train_all, Y_test = train_test_split(X, Y, test_size=0.2, random_state=821)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_all, Y_train_all, test_size=0.2, random_state=821)

# create and train a six-layer neural network with bank dataset
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))
test_loss, test_acc = model.evaluate(X_test, Y_test)

# calculate the precision rate and recall rate
y_predict = model.predict(X_test)
y_predict_binary = (y_predict > 0.5).astype(int)
precision = precision_score(Y_test, y_predict_binary)
recall = recall_score(Y_test, y_predict_binary)

print(f"test_loss:{test_loss}")
print(f"test_accuracy:{test_acc}")
print(f"Precision rate: {precision}")
print(f"Recall rate:{recall}")

# save the model
model.save('../models/trained_models/bank_model.h5')
print("Bank model has been saved...")

# Test_loss = 0.2576976867194304
# Test_accuracy = 0.9001437425613403
# Precision_rate =0.6758409785932722
# Recall rate = 0.21709233791748528

