# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:12:09 2019

@author: Jikhan Jeong
"""


#### 2019 Fall Binary_logistic_regression_Gradient Descent
## Reference : https://github.com/gilbutITbook/006958

import tensorflow as tf
import numpy as np

data =[[2,0],[4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]
x = [i[0] for i in data]
y = [i[1] for i in data]

a = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))

y_hat = 1/(1+np.e**(a*x+b))


loss_before = np.array(y)*tf.log(y_hat)+(1-np.array(y))*tf.log(1-y_hat)
loss = -tf.reduce_mean(loss_before) ## reduce mean for calculating mean



# rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_hat)))

learning_rate = 0.1
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess: ## Sess setting resource in computer, and this function called graph. run with session.run
    sess.run(tf.global_variables_initializer()) # variable initialization
    for step in range(20001):
        sess.run(gradient_decent)
        if step % 2000==0:
            print("Epoch:%.f, loss=%.04f, coefficient a =%.4f, intercept b =%.4f" 
                  % (step,sess.run(loss), sess.run(a), sess.run(b)))
            