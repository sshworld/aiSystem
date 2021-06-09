import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# path = './datasets/train/'
# fruits = os.listdir(path)

# print(fruits)

# num_classes = len(fruits)
  
# image_w = 224
# image_h = 224
  
# X = []
# Y = []
  
# for idex, categorie in enumerate(fruits):
#     label = [0 for i in range(num_classes)]
#     label[idex] = 1
#     image_dir = path + categorie + '/'
  
#     for top, dir, f in os.walk(image_dir):
#         for filename in f:
#             print(image_dir+filename)
#             img = cv2.imread(image_dir+filename)
#             if type(img) == type(None) :
#                 continue
#             img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
#             X.append(img/256)
#             Y.append(label)
 
# X = np.array(X)
# Y = np.array(Y)
 
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
# xy = (X_train, X_test, Y_train, Y_test)
 
# np.save("./img_data.npy", xy)

dir = './datasets/train/'

categories = os.listdir(dir)
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        food_img = cv2.imread(imgpath,0)
        try:
            food_img = cv2.resize(food_img,(224,224))
            image = np.array(food_img).flatten()
            
            data.append([image,label])
        except Exception as e:
            pass

features = []
labels = []

for image, label in data :
    features.append(image)
    labels.append(label)


x_train, x_test, y_train, y_test = train_test_split(features, labels,  test_size=0.2, random_state=42)

xy = (x_train, x_test, y_train, y_test)
 
np.save("./img_data.npy", xy)