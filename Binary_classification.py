#!/home/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-11-24

import numpy as np
from util import load_training_data
def loss(x_shape, y, y_hat, ):
    """
    do the backward calculate, set the self.loss
    param: self.y, self.y_hat
    return: Nothing
    """
    return -1 / x_shape[1] * np.sum((y * np.log(y_hat+0.0000001) + (1 - y) * np.log((1 - y_hat)+0.00000001)))


class logistic_regression:

    layers = None
    x = None
    y = None
    x_shape = None
    y_shape = None
    w1 = None
    b1 = None
    y_hat = None
    loss = None
    dw = None
    db = None
    dJ = None

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
        print("y_hat :"+str(self.y_hat.shape))
        print("loss :" + str(self.loss))

    def initial_w_b(self):
        """
        initial the self.b1 and self.w1
        param: None
        return Nothing
        """
        self.b1 = 0
        try:
            self.w1 = np.zeros((self.layers, self.x_shape[0]))
        except:
            raise AttributeError("please set layers first")

    def sigmoid(self, input_val):
        """
        sigmoid function
        param: input_val
        return: sigmoid(input_val)
        """
        return 1 / (1 + np.exp(-input_val))

    def forward(self):
        """
        do the forward calculate, set the self.y_hat
        param: self.x, self.w1, self.b1
        return: Nothing
        """
        self.y_hat = self.sigmoid(np.dot(self.w1, self.x)+self.b1)

    def gradient_decent(self):
        self.dw = 1/self.x_shape[1] * np.dot((self.y_hat - self.y), self.x.T)
        self.db = 1/self.x_shape[1] * np.sum(self.y_hat - self.y)

    def train(self, iters=1000, learning_rate=0.01):
        iter_ = 0
        while(iter_<iters):
            iter_ += 1
            self.forward()
            self.loss = loss(self.x_shape, self.y, self.y_hat)
            if iter_ % 100 == 0:
                print("iter: " + str(iter_) + "\tloss: " + str(self.loss))
            self.gradient_decent()
            self.w1 = self.w1 - learning_rate * self.dw
            self.b1 = self.b1 - learning_rate * self.db


if __name__ == "__main__":
    x, y = load_training_data() 
    print(x.shape)
    log_reg = logistic_regression(x, y)
    log_reg.set_layers(10)
    log_reg.initial_w_b()
    log_reg.train()
