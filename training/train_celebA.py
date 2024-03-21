"""
- This file preprocesses the celebA dataset and train the CNN model.
- You can access this dataset from the website
    'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'
"""

import os
import cv2
from tqdm import tqdm

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

image_path = '../datasets/celebA/img_align_celeba/'
label_path = '../datasets/celebA/Anno/list_attr_celeba.txt'
model_path = '../models/trained_models/celeba_model.h5'


# get images
def get_images(start, end, type):
    images = []
    if type == 'train':
        origin_images = x_train
    elif type == 'val':
        origin_images = x_val
    else:
        origin_images = x_test

    for i in range(start, end):
        file_path = os.path.join(image_path, origin_images[i])
        img = cv2.resize(cv2.imread(file_path), (256, 256))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images.append(img)
    return np.array(images)


# images
file_names = os.listdir(image_path)

# gender labels
labels = []
with open(label_path, 'r') as file:
    file.readline()
    file.readline()
    lines = file.readlines()
    for idx, line in enumerate(lines):
        data = [x for x in line.strip().split(' ') if x != ''][1:]
        labels.append(data[20])
labels = np.array([0 if val == '-1' else 1 for val in labels])


# spilt the data into training dataset, validation dataset and test dataset
x_train, x_test, y_train, y_test = train_test_split(file_names, labels, test_size=0.2, random_state=821)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=821)


# construct CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dense(2, activation='softmax')
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# train the model
n_epoch = 10
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


# Test
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


# Test_loss = 0.215640
# Test_accuracy =  0.966133
# Precision_rate = 0.954374
# Recall rate =  0.965244
