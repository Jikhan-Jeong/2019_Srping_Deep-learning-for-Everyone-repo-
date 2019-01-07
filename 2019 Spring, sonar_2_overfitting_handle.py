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


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state=seed)
# test set size is 30%, train set is 70%

model = Sequential()
model.add(Dense(24, input_dim=60, activation = 'relu')) # input and hidden 1 layer
model.add(Dense(10, activation = 'relu')) 
model.add(Dense(1, activation ='sigmoid')) # output layer, sigmoid is used


model.compile(loss ='mean_squared_error', # because it is multicalss problem
              optimizer ='adam',
              metrics=['accuracy'])


model.fit(X_train,Y_train, epochs =200, batch_size= 5) # Excuting the train set

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1])) # Testing set


from keras.models import load_model
model.save('my_sonar_model.h5')

model = load_model('my_sonar_model.h5')

print("\n Total set Accuracy: %.4f" % (model.evaluate(X, Y)[1]))



############### K-fold Cross Validation

from sklearn.model_selection import StratifiedKFold
n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)

accuracy = []

for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation = 'relu')) # input and hidden 1 layer
    model.add(Dense(10, activation = 'relu')) 
    model.add(Dense(1, activation ='sigmoid')) # output layer, sigmoid is used
    model.compile(loss ='mean_squared_error', # because it is multicalss problem
              optimizer ='adam',
              metrics=['accuracy'])
    model.fit(X[train],Y[train], epochs =100, batch_size= 5) # Excuting the train set
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1]) # Test accuracy
    accuracy.append(k_accuracy)
    
print("\n %.f fold accuracy:" % n_fold, accuracy)



