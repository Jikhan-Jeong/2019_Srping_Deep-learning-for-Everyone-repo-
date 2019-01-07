# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:37:45 2019

@author: Jikhan Jeong
"""


#### 2019 Fall Multi_logistic_regression
## Reference : https://github.com/gilbutITbook/006958


import tensorflow as tf
import numpy as np

## setting the seed for duplication
seed =0
np.random.seed(seed)
tf.set_random_seed(seed) 


x = np.array([[2,3],[4,3],[6,5],[8,4],[10,8],[12,9],[14,5]])
y = np.array([0,0,0,1,1,1,1]).reshape(7,1)


X = tf.placeholder(tf.float64, shape=[None,2])
Y = tf.placeholder(tf.float64, shape=[None,1])

a = tf.Variable(tf.random_uniform([2,1],dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1],dtype=tf.float64))


y_hat = tf.sigmoid(tf.matmul(X,a) + b) # matmul is for matrix multiplication


loss_before = Y*tf.log(y_hat)+(1-Y)*tf.log(1-y_hat)
loss = -tf.reduce_mean(loss_before) ## reduce mean for calculating mean



# rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_hat)))

learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y>0.5, dtype=tf.float64)
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

print("step, a1, a2, b, loss")

with tf.Session() as sess: ## Sess setting resource in computer, and this function called graph. run with session.run
    sess.run(tf.global_variables_initializer()) # variable initialization
    
    for i in range(3001):
        a_,b_,loss_,_ = sess.run([a,b,loss, gradient_decent], feed_dict={X:x, Y:y})
        if (i+1) % 300==0:

           print(i+1, a_[0], a_[1], b_, loss_)
            
            