# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:30:10 2019

@author: Jikhan Jeong
"""

## 2019 Fall RMSE
## Reference : https://github.com/gilbutITbook/006958

import numpy as np
import tensorflow as tf

ab=[3,14]

data =[[2,80],[4,99],[6,92],[8,98]]
x = [i[0] for i in data]
y = [i[1] for i in data]


def predict(x):
    return ab[0]*x +ab[1]

def rmse(p, a):
    return np.sqrt(((p-a)**2).mean())

def rmse_val(predict_result,y):
    return rmse(np.array(predict_result), np.array(y))

predict_result =[]

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("x =%.f, y=%.f, y_hat=%.f" % (x[i], y[i], predict(x[i])))
    
print("rmse final:" + str(rmse_val(predict_result,y)))
