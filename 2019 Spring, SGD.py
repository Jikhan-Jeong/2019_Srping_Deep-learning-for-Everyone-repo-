# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:22:13 2019

@author: Jikhan Jeong
"""

## 2019 Spring, SGD
## reference : https://github.com/gilbutITbook/006958

# SGD
from keras.datasets import imdb
from keras.utils     import np_utils
from keras.models    import Sequential
import numpy as np

# SGD
self.weight[i] += learning_rate * gradient
keras.optimizers.SGD(lr=0.1)

# SGD with momentum

v = m_rate*v - learning_rate * gradient
self.weight[i] += v
keras.optimizers.SGD(lr=0.1, momentum = 0.9)

# Nesterov momentum

v = m_rate*v - learning_rate * gradient(self.weight[i-1]+m_rate*v)
self.weight[i] += v
keras.optimizers.SGD(lr=0.1, momentum = 0.9, nesterov = True)

# Adagrad, Adaptive Gradient
g += gradient**2
self.weight[i] += - learning_rate*gradient/(np.sqrt(g)+e)
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)

## Rmsprop

g= gamma *g + (1-gamma) * gradient**2
self.weight[i] += - learning_rate*gradient/(np.sqrt(g)+e)
keras.optimizers.RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay =0.0)

## Adam
v = gamma1*m + (1-gamma1)*dx
g = gamma2*v + (1-gamma2)*(dx**2)
x += - learning_rate * g/(np.sqrt(v) + e)
keras.optimizers.Adagrad(lr = 0.01, epsilon =1e-08, decay = 0.0)
