# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 18:59:51 2019

@author: Jikhan Jeong
"""


#### 2019 Fall Multi_regression_with_Gradient Descent
## Reference : https://github.com/gilbutITbook/006958

import tensorflow as tf

data =[[2,0,80],[4,4,99],[6,3,92],[8,5,98]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

a1 = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
a2 = tf.Variable(tf.random_uniform([1],0,10,dtype=tf.float64,seed=0))
b = tf.Variable(tf.random_uniform([1],0,100,dtype=tf.float64,seed=0))

y_hat = a1*x1 + a2*x2 + b

rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_hat)))
learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess: ## Sess setting resource in computer, and this function called graph. run with session.run
    sess.run(tf.global_variables_initializer()) # variable initialization
    for step in range(2001):
        sess.run(gradient_decent)
        if step % 200==0:
            print("Epoch:%.f, RMSE=%.04f, coefficient a1 =%.4f, coefficient a2 =%.4f, intercept b =%.4f" 
                  % (step,sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))