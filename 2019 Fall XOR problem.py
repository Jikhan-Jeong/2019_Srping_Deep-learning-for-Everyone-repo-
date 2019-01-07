# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:42:50 2019

@author: Jikhan Jeong
"""

#### 2019 Fall XOR problem
#### Reference : https://github.com/gilbutITbook/006958

import numpy as np

w11 = np.array([-2,-2]) # Nand gate parameters
b1 = 3

w12 = np.array([2,2])  # Or gate parameters
b2 = -1

w2  = np.array([1,1])  # And gate parameters
b3 = -1

# Perceptron

def perceptron(x,w,b):
    y= np.sum(w*x) +b
    if y<=0:
        return 0
    else:
        return 1

# Non-and Gate
        
def NAND(x1, x2):
    return perceptron(np.array([x1, x2]), w11, b1)

# OR Gate
def OR(x1, x2):
    return perceptron(np.array([x1, x2]), w12, b2)

# And Gate
    
def AND(x1, x2):
    return perceptron(np.array([x1, x2]), w2, b3)

# XOR Gate
    
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

if __name__ == '__main__':
    for x in [(0,0),(1,0),(0,1),(1,1)]:
        y = XOR(x[0],x[1])
        print(x,y)


