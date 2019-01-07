# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:51:50 2019

@author: Jikhan Jeong
"""

## 2019 Spring, Wine
## reference : https://github.com/gilbutITbook/006958

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

## Setting for handling randomness to generate the same results in each trials
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


df = pd.read_csv("C:/python/a_python/2019_Spring_Deep_learning/dataset/wine.csv",
                  header=None)

print(df.info())
print(df.head(3))


dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12] # string values for classes, 3 classes


# test set size is 30%, train set is 70%

model = Sequential()
model.add(Dense(30, input_dim=12, activation = 'relu')) # input and hidden 1 layer
model.add(Dense(12, activation = 'relu')) 
model.add(Dense(8, activation = 'relu')) 
model.add(Dense(1, activation ='sigmoid')) # output layer, sigmoid is used


model.compile(loss ='binary_crossentropy', # because it is multicalss problem
              optimizer ='adam',
              metrics=['accuracy'])


## Early stop setting to stop when training error is not decrease after some wait

early_stopping_callback = EarlyStopping(monitor='val_loss',patience=100)


model.fit(X,Y, epochs =2000, batch_size= 500, callbacks = [early_stopping_callback]) # Excuting the train set

print("\n Test Accuracy: %.4f" % (model.evaluate(X,Y)[1])) # Testing set


history = model.fit(X,Y, validation_split=0.33, epochs = 3500, batch_size=500)
## 33% of all set is used as a test set

y_test_loss = history.history['val_loss'] # test set error
y_acc = history.history['acc']

x_len = numpy.arange(len(y_acc))
plt.plot(x_len, y_test_loss,"o",c="red",markersize =3) # test error red
plt.plot(x_len, y_acc,"o",c="blue",markersize =3)      # accuracy blue




