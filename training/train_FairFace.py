"""
- This file preprocesses the FairFace dataset and train the CNN model.
- You can access this dataset from the website
    'https://github.com/joojs/fairface'
"""

import os
import cv2
from tqdm import tqdm

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


image_path = '../datasets/FairFace/'
label_path = '../datasets/FairFace/'
model_path = '../models/trained_models/FairFace_model.h5'


# get images
def get_images(start, end, type):
    images = []
    d = 'train/'
    if type == 'train':
        origin_images = x_train
    elif type == 'val':
        origin_images = x_val
    else:
        origin_images = x_test
        d = 'val/'

    for i in range(start, end):
        file_path = os.path.join(image_path + d, origin_images[i])
        img = cv2.imread(file_path)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images.append(img)
    return np.array(images)


# preprocess labels
train_df = pd.read_csv(label_path + 'fairface_label_train.csv', sep=',')
test_df = pd.read_csv(label_path + 'fairface_label_val.csv', sep=',')


train_df['gender'] = pd.factorize(train_df['gender'])[0]
test_df['gender'] = pd.factorize(test_df['gender'])[0]

train_labels = train_df['gender'].values
test_labels = test_df['gender'].values


# preprocess images
train_file_names = np.array(sorted(os.listdir(image_path + '/train/'), key=lambda x: int(x.split('.')[0])))
test_file_names = np.array(sorted(os.listdir(image_path + '/val/'), key=lambda x: int(x.split('.')[0])))


# spilt the data into training dataset, validation dataset and test dataset
x_train, x_val, y_train, y_val = train_test_split(train_file_names, train_labels, test_size=0.2, random_state=821)
x_test = test_file_names
y_test = test_labels


# construct CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# # train the model
n_epoch = 35
batch_size = 64
for epoch in range(n_epoch):
    print("====epoch %d=====" % epoch)

    train_loss, train_acc = 0.0, 0.0
    n_batch = len(x_train) // batch_size
    for i in tqdm(range(n_batch), desc="Training Progress"):
        start = i * batch_size
        end = (i + 1) * batch_size
        x_batch = get_images(start, end, 'train')
        y_batch = y_train[start: end].reshape(batch_size, 1)

        loss, acc = model.train_on_batch(x_batch, y_batch)
        train_loss += loss
        train_acc += acc
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc = 0.0, 0.0
    n_batch = len(x_val) // batch_size
    for i in tqdm(range(n_batch), desc="Validating Progress"):
        start = i * batch_size
        end = (i + 1) * batch_size
        x_batch = get_images(start, end, 'val')
        y_batch = y_val[start: end]
        loss, acc = model.test_on_batch(x_batch, y_batch)
        val_loss += loss
        val_acc += acc

    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))

# save the model
model.save(model_path)


# # Test
from keras.models import load_model
model = load_model(model_path)
test_loss, test_acc = 0.0, 0.0
precision, recall = 0.0, 0.0
n_batch = len(x_test) // batch_size
for i in tqdm(range(n_batch), desc="Testing Progress"):
    start = i * batch_size
    end = (i + 1) * batch_size
    x_batch = get_images(start, end, 'test')
    y_batch = y_test[start: end]
    loss, acc = model.evaluate(x_batch, y_batch)
    y_predict = model.predict(x_batch)
    y_predict_binary = (y_predict > 0.5).astype(int)
    precision += precision_score(y_batch, y_predict_binary)
    recall += recall_score(y_batch, y_predict_binary)
    test_loss += loss
    test_acc += acc


print("   Test loss: %f" % (test_loss / n_batch))
print("   Test acc: %f" % (test_acc / n_batch))
print("   Precision_rate: %f" % (precision / n_batch))
print("   Recall rate: %f" % (recall / n_batch))


# Test_loss =  0.710263
# Test_accuracy = 0.809211
# Precision_rate = 0.788966
# Recall rate =  0.810093
