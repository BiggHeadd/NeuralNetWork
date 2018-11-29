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
    w = None
    b = None

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
    
    def initial_param(self, hidden_layers=3):
        """
        initial the param w, b considering with the hidder_layers' numbers
        param: hidden_layers
        return: Nothing
        """
        self.w = np.random.randn(self.x_shape[0], hidden_layers)
        self.b = np.random.randn()
        print("initial param successfully!")

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
    log_reg_more.train()
