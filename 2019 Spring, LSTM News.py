# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:29:13 2019

@author: Jikhan Jeong
"""


## 2019 Spring, LSTM News
## reference : https://github.com/gilbutITbook/006958



from keras.datasets import reuters
from keras.utils     import np_utils
from keras.models    import Sequential
from keras.layers    import Dense,LSTM, Embedding


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

## Setting for handling randomness to generate the same results in each trials
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed) 


(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)
# 20% of data is test set
# num_words = 1000 means the frequency of workd in rank 1000
category = np.max(Y_train)+1
print(category, 'category')
print(len(X_train),'training news')
print(len(X_test),'test news')
print(X_train[0]) # word is indicated as a number


from keras.preprocessing import sequence
x_train = sequence.pad_sequences(X_train, maxlen = 100)
x_test  = sequence.pad_sequences(X_test, maxlen = 100)

y_train = np_utils.to_categorical(Y_train)
y_test  = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Embedding(1000,100)) # tansfering function, 1000 total word, 100 words in each articls
model.add(LSTM(100, activation = 'tanh')) # Handling the weight on the lag variables, 100words in each articles
model.add(Dense(46, activation = 'softmax')) # output 46 categories
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer ='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 20, 
                    batch_size = 100)

print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1])) # .4f means 4 decimal percision



y_test_loss  = history.history['val_loss'] # test  set error
y_train_loss = history.history['loss']     # train set error

x_len = np.arange(len(y_train_loss)) # y_train_loss = 19, x_len = arrange(19)

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_len, y_test_loss,  marker='.', c="red" , label ='Test_error')
plt.plot(x_len, y_train_loss, marker='.', c="blue", label ='Train_error')
plt.show()