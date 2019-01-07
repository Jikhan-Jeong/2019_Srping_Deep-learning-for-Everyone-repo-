# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:51:50 2019

@author: Jikhan Jeong
"""

## 2019 Spring, Mulit_calss_Iris
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

df = pd.read_csv("C:/python/a_python/2019_Spring_Deep_learning/dataset/iris.csv",
                 names = ["sepal_length","sepal_width","petal_length","petal_width","species"])

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='species');
plt.show()

dataset = df.values
X = dataset[:,0:4].astype(float)
Y_obj = dataset[:,4] # string values for classes, 3 classes
type(Y_obj)

from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj) # string to numerical 

from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y) # one-hot encoding

model = Sequential()
model.add(Dense(16, input_dim=4, activation = 'relu')) # input and hidden 1 layer
model.add(Dense(3, activation ='softmax')) # output layer, softmax is used

model.compile(loss ='categorical_crossentropy', # because it is multicalss problem
              optimizer ='adam',
              metrics=['accuracy'])

model.fit(X,Y_encoded, epochs =50, batch_size= 1) # Excuting the model

print("\n Accuracy: %.4f" % (model.evaluate(X,Y_encoded)[1]))

