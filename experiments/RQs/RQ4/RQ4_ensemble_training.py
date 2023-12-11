"""
- RQ4
- This file use the ensemble models for relabeling.
- We use five classifiers are trained for majority voting, including KNN, Naive Bayes, MLP, Random Forest, RBF SVM.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


