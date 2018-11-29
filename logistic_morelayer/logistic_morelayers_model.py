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

if __name__ == "__main__":
    x, y = load_training_data()
    log_reg_more = logistic_regression_more(x, y)
    sigmoid = log_reg_more.sigmoid(x)
    print(sigmoid.shape)
