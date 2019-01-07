# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 18:47:09 2019

@author: Jikhan Jeong
"""

#### 2019 Fall Gradient Descent
## Reference : https://github.com/gilbutITbook/006958

import tensorflow as tf

data =[[2,80],[4,99],[6,92],[8,98]]
x = [i[0] for i in data]
y = [i[1] for i in data]

a = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))

y_hat = a*x + b

rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_hat)))
learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess: ## Sess setting resource in computer, and this function called graph. run with session.run
    sess.run(tf.global_variables_initializer()) # variable initialization
    for step in range(2001):
        sess.run(gradient_decent)
        if step % 200==0:
            print("Epoch:%.f, RMSE=%.04f, coefficient a =%.4f, intercept b =%.4f" 
                  % (step,sess.run(rmse), sess.run(a), sess.run(b)))
            