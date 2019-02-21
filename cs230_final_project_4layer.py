
# coding: utf-8

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from glob import glob
import ast
import pandas as pd
import json
import cv2
from PIL import Image
from keras import Sequential


# In[3]:


#data cleaning
size = 32
n_classes = 340
img_per_class = 2000

def cv2array(strokes):
    img = np.zeros((256, 256), np.uint8) + 255
    for i, stroke in enumerate(strokes):
        for j in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][j], stroke[1][j]),
                         (stroke[0][j + 1], stroke[1][j + 1]), 0, 6)
    return cv2.resize(img, (size, size))

train_data = []
label_data = []
paths = glob("../train_simplified/*")

#df = pd.read_csv(paths[0], nrows = img_per_class)


# In[4]:


np.random.seed(0)
for i, path in enumerate(paths):
    df = pd.read_csv(path, nrows = img_per_class)
    df["drawing"]= df["drawing"].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    y = np.zeros((len(df), n_classes))
    for j, vec in enumerate(df.drawing.values):
        x[j, :, :, 0] = cv2array(vec)/255.0
        y[j, i] = 1
    label_data.append(y)
    train_data.append(x)
x_train = np.concatenate(train_data, axis = 0)
y_train = np.concatenate(label_data, axis = 0)

print("data loaded.")
# In[5]:


del train_data
del label_data
#img = Image.fromarray(x_train[201, :, :, 0])   plot image
#img.show()

#shuffle data
randomize = np.arange(len(x_train))
np.random.shuffle(randomize)
x_train = x_train[randomize]
y_train = y_train[randomize]

dev_set_ratio = 0.01
dev_set = int(dev_set_ratio * x_train.shape[0])
print("shuffle")

#split
x_dev = x_train[:dev_set]
y_dev = y_train[:dev_set]
x_train = x_train[dev_set:]
y_train = y_train[dev_set:]

input_shape = (size, size, 1)


# In[6]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[7]:


# CNN Model
# 2-layer CNN
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())

model2.add(Dense(680, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(340, activation='softmax'))

model2.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# model parameter check
model2.summary()


# In[ ]:


#model fitting
result2 = model2.fit(x=x_train, y=y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_dev, y_dev))

#model saving
model2.saving('4layer.h4')
