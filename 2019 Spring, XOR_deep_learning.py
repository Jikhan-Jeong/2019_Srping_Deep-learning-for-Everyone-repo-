# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:06:45 2019

@author: Jikhan Jeong
"""

## 2019 Spring, XOR_deep_learning
## reference : https://github.com/gilbutITbook/006958
# Reference: http://arctrix.com/nas/python/bpnn.py (Neil Schemenauer)


import random
import numpy as np

random.seed(777)

data = [
        [[0,0],[0]],
        [[0,1],[1]],
        [[1,0],[1]],
        [[1,1],[0]],
        ]

iterations = 5000   # iteration
learning_rate = 0.1 # learning rate
mo = 0.9            # momenterm

# Activation Function : sigmoid
def sigmoid(x, derivative = False):
    if (derivative == True):
        return x*(1-x)      # derevative
    return 1/(1+np.exp(-x))

# Activation Function : tanh
    
def tanh(x, derivative = False):
    if (derivative == True):
        return 1-x**2       # derivative
    return np.tanh(x)

def makeMatrix(i, j, fill = 0.0): # making weighting matrix
    mat = []
    for i in range(i):
        mat.append([fill]*j)
    return mat

class NeuralNetwork:
    
    # Initial Values
    
    def __init__(self, num_x, num_yh, num_yo, bias = 1): # input, hidden, output, bias
        self.num_x  = num_x + bias
        self.num_yh = num_yh
        self.num_yo = num_yo
        
    # Activation initial values    
        self.activation_input =[1.0]*self.num_x
        self.activation_hidden=[1.0]*self.num_yh
        self.activation_out   =[1.0]*self.num_yo
        
    # weighting input initial values
    
        self.weight_in = makeMatrix(self.num_x, self.num_yh)
        for i in range(self.num_x):
            for j in range(self.num_yh):
                self.weight_in[i][j] = random.random()
                
    # Weiting output initial value
    
        self.weight_out = makeMatrix(self.num_yh, self.num_yo)
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()
                
    # Momentum SGD initial value

        self.gradient_in  = makeMatrix(self.num_x, self.num_yh) # input, hidden
        self.gradient_out = makeMatrix(self.num_yh, self.num_yo)# hidden, output             

    # Updata function
    
    def update(self, inputs):
        
        # input activation
        
        for i in range(self.num_x -1):
            self.activation_input[i] = inputs[i]
            
        # hidden activation
        
        for j in range(self.num_yh):
            sum = 0.0
            for i in range(self.num_x):
                sum = sum + self.activation_input[i] * self.weight_in[i][j]
            self.activation_hidden[j] = tanh(sum, False)    
        # output activation
        
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh):
                sum = sum + self.activation_hidden[j] * self.weight_out[j][k]
            self.activation_out[k] =  tanh(sum, False)          
        return self.activation_out[:]
    
    # BackPropagate
    
    def backPropagate(self, targets):
        
        # delta
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            output_deltas[k] = tanh(self.activation_out[k], True)*error
            
        hidden_deltas =[0.0]*self.num_yh
        for j in range(self.num_yh):
            error = 0.0
            for k in range(self.num_yo):
                error = error + output_deltas[k]* self.weight_out[j][k]
            hidden_deltas[j] = tanh(self.activation_hidden[j], True)*error
        
        # output weight update
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                gradient = output_deltas[k] * self.activation_hidden[j]
                v = mo*self.gradient_in[j][k] - learning_rate*gradient
                self.weight_in[j][k]   += v
                self.gradient_out[j][k] = gradient
                
        # input weight update
        for i in range(self.num_x):
            for j in range(self.num_yh):
                gradient = hidden_deltas[j] * self.activation_input[i]
                v = mo*self.gradient_in[i][j] - learning_rate * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient
        
        # error calcuation
        error =0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.activation_out[k])**2
            
        return error
    
    # execution
    
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets= p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 ==0:
                print('error: %-.5f' % error)
                
    def result(self, patterns):
         for p in patterns:
             print('Input: %s, Predict: %s' % (p[0], self.update(p[0])))
    
if __name__ == '__main__':

         n =  NeuralNetwork(2,2,1) # two input, two layers, 1 output 
         
         n.train(data)
         
         n.result(data)
             
        