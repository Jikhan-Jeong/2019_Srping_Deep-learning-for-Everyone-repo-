# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:51:50 2019

@author: Jikhan Jeong
"""


## 2019 Spring, Indian prepareing and analysis
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

df = pd.read_csv("C:/python/a_python/2019_Spring_Deep_learning/dataset/pima-indians-diabetes.csv",
                 names = ["pregnant","plasma","pressure","thickness","insulin","BMI","pedigree","age","class"])


dataset = numpy.loadtxt("C:/python/a_python/2019_Spring_Deep_learning/dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]


print(df.head(5))
print(df.info())
print(df.describe())
print(df[['pregnant','class']].groupby(['pregnant'], as_index =False).mean().sort_values(by='pregnant', ascending=True))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize =(12,12))
sns.heatmap(df.corr(), linewidths = 0.1, vmax =0.5, cmap = plt.cm.gist_heat, linecolor = 'white', annot= True)
# vmax for brightness
# cmap is color

grid = sns.FacetGrid(df, col ='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

model = Sequential()
model.add(Dense(12, input_dim=8, activation = 'relu')) # input and hidden 1 layer
model.add(Dense(8, activation = 'relu')) # hidden 2 layer
model.add(Dense(1, activation ='sigmoid')) # output layer

model.compile(loss ='binary_crossentropy',
              optimizer ='adam',
              metrics=['accuracy'])

model.fit(X,Y, epochs =200, batch_size= 10)


print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1]))

