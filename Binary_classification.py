#!/home/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-11-24

import numpy as np

class logistic_regression:

    layers = None
    x = None
    y = None
    x_shape = None
    y_shape = None
    w1 = None
    b1 = None

    def __init__(self, x, y):
        """
        get input training x, y samples, and initial the self.x_shape, self.y_shape, self.x, self.y
        param: x, y
        numpy data
        return Nothing
        """
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y

    def set_layers(self, layers):
        """
        get layers and set self.layers
        param: layers
        return Nothing
        """
        self.layers = layers

    def print_all(self):
        """
        print the self values
        param: None
        return Nothing
        """
        print("layers: "+str(self.layers))
        print("x_shape: "+str(self.x_shape))
        print("y_shape: "+str(self.y_shape))
        print("w1 :"+str(self.w1.shape))
        print("b1 :"+str(self.b1))

    def initial_w_b(self):
        """
        initial the self.b1 and self.w1
        param: None
        return Nothing
        """
        self.b1 = 0
        try:
            self.w1 = np.zeros((1, self. layers))
        except:
            raise AttributeError("please set layers first")



if __name__ == "__main__":
    x = np.ones((10, 300))
    y = np.ones((1, 10))
    log_reg = logistic_regression(x, y)
    log_reg.set_layers(10)
    log_reg.initial_w_b()
    log_reg.print_all()

