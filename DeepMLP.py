import cv2
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import time
start = time.time()

x_train, x_test, y_train, y_test = np.load('./img_data.npy', allow_pickle=True)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 10, 7), learning_rate_init=0.0005, batch_size=512, solver='adam',
                    verbose=True, max_iter=100)


mlp.fit(x_train, y_train)
res = mlp.predict(x_test)

print(len(res))

conf = np.zeros((36, 36))

for i in range(len(res)) :
    conf[res[i]][y_test[i]] += 1
print(conf)

correct = 0
for i in range(36) :
    correct += conf[i][i]
accuracy = correct / len(res)
print(f'Accuracy on the test set: {100*accuracy:.2f}%')

print("time : ", time.time() - start)