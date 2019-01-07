# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:17:40 2019

@author: Jikhan Jeong
"""

## 2019 Spring, Deep Learning
## reference : https://github.com/gilbutITbook/006958


from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow as tf


## Setting for handling randomness to generate the same results in each trials
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

Data_set = numpy.loadtxt("C:/python/a_python/2019_Spring_Deep_learning/dataset/ThoraricSurgery.csv",delimiter=",")



X = Data_set[:,0:17]
Y = Data_set[:,17]
# X.shape
# Y.shape

## Deeplearning setting
model = Sequential() # building hidden layer
model.add(Dense(30, input_dim=17, activation='relu')) ## dense 30 means node =30
model.add(Dense(1, activation='sigmoid')) ## output layer 1 node as a output

## Deeplearning excution
model.compile(loss='mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X,Y, epochs=30, batch_size=10)


## Printing the results
print("/n Accurarcy: %.4f" % (model.evaluate(X,Y)[1]))
