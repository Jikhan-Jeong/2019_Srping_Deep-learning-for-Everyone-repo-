# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:00:18 2019

@author: Jikhan Jeong
"""



## 2019 Spring, LSTM Movie imdb
## reference : https://github.com/gilbutITbook/006958


from keras.datasets import imdb
from keras.utils     import np_utils
from keras.models    import Sequential
from keras.layers    import Dense,LSTM, Embedding,Dropout, Activation, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

## Setting for handling randomness to generate the same results in each trials
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed) 

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
# num_words = 5000 means the frequency of workd in rank 1000

from keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train, maxlen = 100)
x_test  = sequence.pad_sequences(x_test, maxlen = 100)

model = Sequential()
model.add(Embedding(5000,100)) # tansfering function, 1000 total word, 100 words in each articls
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding ='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55)) # Handling the weight on the lag variables, 100words in each articles
model.add(Dense(1)) # output 46 categories
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer ='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 3, 
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