# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 22:13:17 2019

@author: Jikhan Jeong
"""

## 2019 Spring, housing_regression
## reference : https://github.com/gilbutITbook/006958


import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

## Setting for handling randomness to generate the same results in each trials
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("C:/python/a_python/2019_Spring_Deep_learning/dataset/housing.csv",
                  delim_whitespace=True, header=None)

print(df.info())
df.shape

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13] # string values for classes, 3 classes

X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,test_size=0.3, random_state=seed)


# test set size is 30%, train set is 70%

model = Sequential()
model.add(Dense(30, input_dim=13, activation = 'relu')) # input and hidden 1 layer
model.add(Dense(6, activation = 'relu')) 
model.add(Dense(1)) # output layer, sigmoid is used


model.compile(loss ='mean_squared_error', # because it is multicalss problem
              optimizer ='adam',
              metrics=['accuracy'])

model.fit(X_train,Y_train, epochs =200, batch_size= 10)

Y_prediction = model.predict(X_test).flatten()

for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("Y:{:.3f}, Y_hat:{:.3f}".format(label, prediction))





