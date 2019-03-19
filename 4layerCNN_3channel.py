import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.models import Sequential
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Flatten
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from glob import glob
import ast
import pandas as pd
import json
import cv2
from PIL import Image

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

df = pd.read_csv(paths[0], nrows = img_per_class)


for i, path in enumerate(paths):
    df = pd.read_csv(path, usecols = ["drawing", "recognized"], nrows = img_per_class * 5//4)
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
#print(x_train.shape)
#plt.imshow(x_train[1,:,:,:])
#plt.show()

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

input_shape = (size, size, 3)


# CNN Model
# 4-layer CNN
model_cnn4 = Sequential()
model_cnn4.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model_cnn4.add(MaxPooling2D((2, 2)))
model_cnn4.add(Dropout(0.1))

model_cnn4.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model_cnn4.add(MaxPooling2D((2, 2)))
model_cnn4.add(Dropout(0.1))

model_cnn4.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model_cnn4.add(MaxPooling2D((2, 2)))
model_cnn4.add(Dropout(0.1))

model_cnn4.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model_cnn4.add(MaxPooling2D((2, 2)))
model_cnn4.add(Dropout(0.1))

model_cnn4.add(Flatten())

model_cnn4.add(Dense(16800, activation='relu'))
model_cnn4.add(Dropout(0.1))
model_cnn4.add(Dense(340, activation='softmax'))

model_cnn4.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# model parameter check
model_cnn4.summary()

#model fitting
result_cnn4 = model_cnn4.fit(x=x_train, y=y_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_dev, y_dev))

#parameter saving
model_cnn4.save('model_cnn4_3channel.h5')
model_cnn4.save_weights("model_cnn4_3channel_weight.h5")
pd.DataFrame(result_cnn4.history).to_csv("history_model_cnn4_3channel.csv")

