# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:30:12 2019

@author: Jikhan Jeong
"""

## 2019/01/03 Linear Regression
## reference : https://github.com/gilbutITbook/006958

import numpy as np
x = [2,4,6,8]
y = [180,200,230,240]

mx = np.mean(x)
my = np.mean(y)

denomirator = sum([(mx - i)**2 for i in x])

def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx)*(y[i]-my)
    return d

numerator = top(x, mx, y, my)

a = numerator/denomirator
b = my -(mx*a)

print("coefficient: ",a)
print(" intercept:", b)
