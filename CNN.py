import os
from PIL import Image
import numpy as np
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from cv2 import cv2
import matplotlib.pyplot as plt


path = './datasets/train/'
fruits = []
for x in os.listdir(path):
    fruits.append(x)

data=[]
labels=[]
im_w = 224
im_h = 224

for x in range(len(fruits)):
    sub_path = path+fruits[x]+'/'
    for y in os.listdir(sub_path):        
        img_path = sub_path+y  
        last = img_path[-12:]
        imag=cv2.imread(img_path)  
        if last == 'Image_56.jpg':
            continue
        if last == 'Image_96.jpg': 
            continue
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((im_w, im_h))
        data.append(np.array(resized_image))
        labels.append(x)

categories=np.array(data)
labels=np.array(labels)

s=np.arange(categories.shape[0])
np.random.shuffle(s)
categories=categories[s]
labels=labels[s]

num_classes=len(np.unique(labels))
data_length=len(categories)

(x_train,x_test)=categories[(int)(0.1*data_length):],categories[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(im_w,im_h,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(36, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=17)
history = model.fit(x_train,y_train,batch_size=50, epochs=90,verbose=1, validation_split=0.33, callbacks=[early_stop])

score = model.evaluate(x_test, y_test, verbose=1)
print(f'Accuracy on the test set: {100*score[1]:.2f}%')