# !/home/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-11-29

import numpy as np
import os
from util import load_training_data

class logistic_regression_more:

    x_shape = None
    y_shape = None
    x = None
    y = None
    hidden_layers = None
    param_cache = list()
    A_cache = list()

    def __init__(self, x, y):
        """
        set the self's x, y, x_shape, y_shape by input x, y
        param: x, y(both np.array)
        return None
        """
        self.x = x
        self.y = y
        self.x_shape = x.shape
        self.y_shape = y.shape

    def sigmoid(self, z):
        """
        calculate the sigmoid function
        param: z (real number or np.array)
        return: sigmoid(z)
        """
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid
    
    def initial_param(self, layer_l_n_1, layer_l_a_1):
        """
        initial the param w, b considering with the hidder_layers' numbers, and append to the list self.cache, w and b save in the dict A, the key is "w" and "b"
        param: layer_l_n_1(A[l-1]), layer_l_a_1(A[l])
        return: w, b (both np.arrays)   w.shape -> (A[l], A[l-1])
        """
        w = np.random.randn(layer_l_a_1, layer_l_n_1)
        b = np.random.randn()
        param_cache_tmp = dict()
        param_cache_tmp['w'] = w
        param_cache_tmp['b'] = b
        self.param_cache.append(param_cache_tmp)
        print("initial param successfully!")
        return w, b

    def forward_propergation(self, A, A_n_1, A_a_1, activation_type="sigmoid"):
        """
        initial the param using the initial_param(), and do the forward propergation, the activation is the "activation_type", and add to the cache
        param: A(the input, like x), A_n_1(A[l-1]), A_a_1(A[l]), activation_type(String, name of activation)
        return: None
        """
        w, b = self.initial_param(A_n_1, A_a_1)
        z = np.dot(w, A) + b
        if activation_type == "sigmoid":
            A = self.sigmoid(z)
        A_cache_tmp = dict()
        A_cache_tmp['A'] = A
        self.A_cache.append(A_cache_tmp)
        print(A_cache_tmp['A'].shape)


    def train(self, hidden_layers=3, iters=1):
        self.hidden_layers = hidden_layers
        self.initial_param()
        i = 0
        while(i<iters):
            print(self.w.shape)
            print(self.x_shape)
            z = np.dot(self.w.T, self.x) + self.b
            a = self.sigmoid(z)
            i+=1
        print(a.shape)


if __name__ == "__main__":
    x, y = load_training_data()
    log_reg_more = logistic_regression_more(x, y)
    w, b = log_reg_more.initial_param(2, 3)
    print("w: " + str(w.shape) + "\nw_shape: " + str(w))
    print("b: " + str(b))
    log_reg_more.forward_propergation(x, 2, 3, "sigmoid")
