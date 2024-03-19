"""
- This file preprocesses the celebA dataset and train the CNN model.
- You can access this dataset from the website
    'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'
"""

import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


data_path = '../datasets/celebA/img_align_celeba/'
label_path = '../datasets/celebA/Anno/list_attr_celeba.txt'
model_path = '../models/trained_models/celeba_model.h5'

images = []
labels = []

# gender labels
with open(label_path, 'r') as file:
    file.readline()
    file.readline()
    lines = file.readlines()
    for idx, line in enumerate(lines):
        data = [x for x in line.strip().split(' ') if x != ''][1:]
        labels.append(data[20])
labels = tf.keras.utils.to_categorical(labels, num_classes=2)
print("Get Labels...")

# images
file_names = os.listdir(data_path)
for file_name in file_names:
    file_path = os.path.join(data_path, file_name)
    img = cv2.resize(cv2.imread(file_path), (256, 256))
    img = normalized_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    images.append(img)

print("Get Images...")

# spilt the data into training dataset, validation dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=821)


# construct CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2表示两个性别类别
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # 多类别交叉熵损失
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# save the model
model.save(model_path)




