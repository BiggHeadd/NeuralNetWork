#!/home/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-11-24

import numpy as np
from util import load_training_data, load_test_data
import os


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
    y_predict = None
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
        """
        calculate the gradient of w and b, and set the class value self.w1 and self.b1
        param: None
        return: Nothing
        """
        self.dw = 1/self.x_shape[1] * np.dot((self.y_hat - self.y), self.x.T)
        self.db = 1/self.x_shape[1] * np.sum(self.y_hat - self.y)

    def train(self, iters=1000, learning_rate=0.01):
        """
        training, set param or it will be default setting
        param: iters, learning_rate
        return Nothing
        """
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

    def print_w1_b1(self):
        """
        just print the w1 and b1 to see what happen
        param: None
        return Nothing
        """
        print("w1: " + str(self.w1))
        print("b1: " + str(self.b1))

    def save_param(self, file_name="default"):
        """
        save the param(w1, b1) into the file (default: "model/default_w1_.model", "model/default_b1_.model"
        param: file_name    the model name (empty for default)
        return: Nothing
        """
        save_dir = "model/"
        with open(os.path.join(save_dir, file_name+"_w1_.model"), 'w', encoding='utf-8')as f:
            string = ""
            for w in self.w1:
                string_tmp = ""
                for each_w in w:
                    string_tmp += str(each_w)+'\t'
                string += string_tmp[:-1]
                string += '\n'
            f.write(string)
        with open(os.path.join(save_dir, file_name+"_b1_.model"), 'w', encoding='utf-8')as f:
            f.write(str(self.b1))
                    
    def load_param(self, file_name="default"):
        load_dir = "model/"
        with open(os.path.join(load_dir, file_name+"_w1_.model"), 'r', encoding='utf-8')as f:
            w_load = list()
            for w in f.readlines():
                w_tmp = list()
                for each_w in w.split('\t'):
                    w_tmp.append(np.float(each_w))
                w_load.append(w_tmp)
            w_load = np.array(w_load)
        with open(os.path.join(load_dir, file_name+'_b1_.model'), 'r', encoding='utf-8')as f:
            b_load = f.read()
        b_load = np.float(b_load)
        
        self.w1 = w_load
        self.b1 = b_load

    def predict(self, test_x):
        """
        get the answer of test_x by calculate logistic using the param learned before, and set the self.y_predict
        param: test_x(np.array)
        return: y_predict(np.array)
        """
        print("predict......")
        y_predict = self.sigmoid(np.dot(self.w1, test_x) + self.b1)
        print("finish.......")
        y_result = list()
        for y in y_predict[0]:
            if y > 0.5:
                y_result.append(1)
            else:
                y_result.append(0)
        self.y_predict = y_result
        return y_result


if __name__ == "__main__":
    x, y = load_training_data() 
    print(x.shape)
    log_reg = logistic_regression(x, y)
    log_reg.load_param()
    test_x = load_test_data()
    y_predict = log_reg.predict(test_x)
    print(y_predict)
