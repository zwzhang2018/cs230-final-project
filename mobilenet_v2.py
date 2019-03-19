#!/usr/bin/env python

# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from glob import glob
import ast
import pandas as pd
import json
import cv2
from PIL import Image

import keras


# In[3]:


#data cleaning
size = 32
n_classes = 340
img_per_class = 3000

def cv2array(strokes):
    img = np.zeros((256, 256, 3), np.uint8)
    for i, stroke in enumerate(strokes):
        for j in range(len(stroke[0]) - 1):
            color = (255, 255-min(i,10)*13, max(255 - len(stroke[0])*j, 20))
            _ = cv2.line(img, (stroke[0][j], stroke[1][j]),
                         (stroke[0][j + 1], stroke[1][j + 1]), color, 6)
    return cv2.resize(img, (size, size));

dev_set_ratio = 0.2


train_data = []
label_data_train = []
dev_data = []
label_data_dev = []
paths = glob("../train_simplified/*")

#df = pd.read_csv(paths[0], nrows = img_per_class)


# In[4]:


for i, path in enumerate(paths):
    df = pd.read_csv(path, usecols = ["drawing", "recognized"], nrows = img_per_class *     5//4)
    df = df[df.recognized == True].head(img_per_class)
    df["drawing"]= df["drawing"].apply(json.loads)
    x = np.zeros((len(df), size, size, 3))
    y = np.zeros((len(df), n_classes))
    for j, vec in enumerate(df.drawing.values):
        x[j, :, :, :] = cv2array(vec)/255.0
        y[j, i] = 1
    dev_number = int(dev_set_ratio * len(x))

    label_data_dev.append(y[:dev_number,:])
    dev_data.append(x[:dev_number,:,:,:])

    label_data_train.append(y[dev_number:,:])
    train_data.append(x[dev_number:,:,:,:])

x_train = np.concatenate(train_data, axis = 0)
y_train = np.concatenate(label_data_train, axis = 0)
x_dev = np.concatenate(dev_data, axis = 0)
y_dev = np.concatenate(label_data_dev, axis = 0)

print("data loaded.")
# In[5]:


del train_data
del label_data_train
del dev_data
del label_data_dev
#img = Image.fromarray(x_train[201, :, :, 0])   plot image
#img.show()

#shuffle train data
randomize_train = np.arange(len(x_train))
np.random.shuffle(randomize_train)
x_train = x_train[randomize_train]
y_train = y_train[randomize_train]

#shhuffle dev data
randomize_dev = np.arange(len(x_dev))
np.random.shuffle(randomize_dev)
x_dev = x_dev[randomize_dev]
y_dev = y_dev[randomize_dev]

print("Shuffle")


# In[6]:

input_shape = (size, size, 1)

# In[7]:


# MobileNet2 Model
model2=keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, alpha=1.0, weights=None, pooling='max', classes=340)

model2.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# model parameter check
model2.summary()


# In[9]i:

#model fitting
result2 = model2.fit(x=x_train, y=y_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_dev, y_dev))

#model saving
model2.save_weights('mobilenetv2_6_weight_lr.h5')
model2.save('mobilenetv2_6_lr.h5')

#result saving
pd.DataFrame(result2.history).to_csv("rgb_32_lr.csv")





