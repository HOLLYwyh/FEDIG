"""
- This file constructs training set, test set.
- This file also trains the CNN model of celebA Dataset
"""

import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

data_path = '../datasets/celebA/img_align_celeba'








