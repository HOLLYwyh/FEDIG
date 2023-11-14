import numpy as np
import pandas as pd
import ADF
import DICE
import EIDIG
import NeuronFair
import tensorflow as tf
from tensorflow import keras
from utils import config
from tensorflow.keras.models import load_model, Model

path = '../models/trained_models/credit_model.h5'
data_path = '../datasets/credit'

df = pd.read_csv(data_path)
data = df.values[:, :-1]
x = data


model = load_model(path)


# new_model_input = keras.Input(model.layers[0].input_shape)
# x_t = new_model_input
# for layer in model.layers[:-1]:  # 提取前5层
#     x_t = layer(x_t)
#
# # 创建新模型
# new_model = keras.Model(inputs=new_model_input, outputs=x_t)
#
# x1 = x[0]
# x2 = x[1]
# neurons = np.array([0, 0, 1, 1])
# r = np.array([0, 0, 0, 0])
#
#
# x1 = tf.constant([x1], dtype=tf.float32)
# x2 = tf.constant([x2], dtype=tf.float32)
# p = neurons | r
# p = tf.constant(p, dtype=tf.float32)
#
#
#
# with tf.GradientTape() as tape:
#     tape.watch(x1)
#     y_predict_x1 = tf.nn.sigmoid(new_model(x1))
#     y_predict_x2 = new_model(x2)
#     y_predict = -1 * tf.reduce_sum(p * y_predict_x2 * y_predict_x1)
#
#
# gradient = tape.gradient(y_predict, x1)
#
# res_grad = 1 / (abs(gradient) + 1e-6)
# res = tf.nn.softmax(res_grad)
# print(res)




# test for NeuronFair
# all_id = NeuronFair.individual_discrimination_generation('credit', config.Credit, model)
# print(len(all_id))
# print(all_id)

# test for ADF
# all_id = ADF.individual_discrimination_generation('credit', config.Credit, model)
# all_id = EIDIG.individual_discrimination_generation('credit', config.Credit, model)
all_id = DICE.individual_discrimination_generation('credit', config.Credit, model)
print(all_id)
print(all_id.shape)

