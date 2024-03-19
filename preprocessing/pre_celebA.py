"""
- This file preprocesses the celebA dataset.
- You can access this dataset from the website
    'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'
"""

import os
import cv2
import numpy as np
import tensorflow as tf


data_path = '../datasets/celebA/img_align_celeba/'

images = []

file_names = os.listdir(data_path)

for file_name in file_names:
    file_path = os.path.join(data_path, file_name)
    img = cv2.imread(file_path)
    images.append(img)

print(len(images))





