# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:51:50 2019

@author: Jikhan Jeong
"""

## 2019 Spring, Sonar_Overfitting
## reference : https://github.com/gilbutITbook/006958

from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow as tf
import pandas as pd

## Setting for handling randomness to generate the same results in each trials
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("C:/python/a_python/2019_Spring_Deep_learning/dataset/sonar.csv",
                  header=None)
print(df.info())
print(df.head(3))


dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60] # string values for classes, 3 classes
type(Y_obj)

from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj) # string to numerical 


model = Sequential()
model.add(Dense(24, input_dim=60, activation = 'relu')) # input and hidden 1 layer
model.add(Dense(10, activation = 'relu')) 
model.add(Dense(1, activation ='sigmoid')) # output layer, sigmoid is used


model.compile(loss ='mean_squared_error', # because it is multicalss problem
              optimizer ='adam',
              metrics=['accuracy'])


model.fit(X,Y, epochs =200, batch_size= 5) # Excuting the model

print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))

