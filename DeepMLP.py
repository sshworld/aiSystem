import cv2
import numpy as np
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import time
start = time.time()

inputLength = 224 * 224

x_train, x_test, y_train, y_test = np.load('./img_data.npy', allow_pickle=True)

x_train = np.array(x_train).reshape(len(x_train), inputLength)
x_test = np.array(x_test).reshape(len(x_test), inputLength)
x_train = x_train/255.0
x_test = x_test/255.0
y_train = tf.keras.utils.to_categorical(y_train, 36)
y_test = tf.keras.utils.to_categorical(y_test, 36)

n_input = 224 * 224
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_hidden4 = 64
n_output = 36

mlp = Sequential()
mlp.add(Dense(units = n_hidden1, activation = 'tanh',
              input_shape=(n_input,), kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
mlp.add(Dense(units = n_hidden2, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_hidden3, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_hidden4, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))
mlp.add(Dense(units = n_output, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer ='zeros'))

mlp.compile(loss = 'mse', optimizer = 'sgd', metrics = ['accuracy'])
hist = mlp.fit(x_train, y_train, batch_size = 128, epochs = 100 , validation_data = (x_test,y_test), verbose = 2)

res = mlp.evaluate(x_test, y_test, verbose = 0)

print(f'Accuracy on the test set: {int(res[1]*100):.2f}%')

print("time : ", time.time() - start)