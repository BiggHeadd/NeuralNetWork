# !/home/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-12-3

import numpy as np
from util import load_training_data

class logistic_regression():

    parameters = None
    cache = None

    def __init__(self):
        pass

    def get_shape(self, X, Y):
        """
        get the feature of the x and y, return and store in the member variables
        Arguments:
        X -- the input of training set
        Y -- the label of training set

        Returns:
        n_x -- the feature number of the input data
        n_y -- the feature number of the output data
        """
        n_x = x.shape[0]
        n_y = y.shape[0]
        return n_x, n_y

    def sigmoid(self, z):
        """
        the sigmoid function
        Argument:
        z -- the data after the linear function

        Return:
        sigmoid -- sigmoid(z)
        """
        sigmoid = 1 / (1+np.exp(-z))
        return sigmoid

    def initial_parameters(self, n_x, n_y, n_h=4):
        """
        initial the parameters, save them into a dictionary
        Arguments:
        n_x -- the feature number of the input data
        n_y -- the feature number of the ouput data
        n_h -- the cell number of the hidden layer 

        Returns:
        Nothing
        """
        W1 = np.random.randn(n_h, n_x)
        b1 = np.random.randn(n_h, 1)
        W2 = np.random.randn(n_y, n_h)
        b2 = np.random.randn(n_y, 1)

        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))

        parameters = {"W1":W1,
                      "b1":b1,
                      "W2":W2,
                      "b2":b2}
        self.parameters = parameters
        return


    def forward_propagation(self, X, Y, n_h=4, activation='sigmoid'):
        """
        calculate the forward propagation, save each layers' cell number that through the calculating
        Arguments:
        X -- the inputs of the training set
        Y -- the labels of the training set
        parameters -- the python dictionary containing the params (w b and so)
        n_h -- the cells number of the hidden layers
        activation -- the type of the activation function

        Returns:
        A2 -- the output of the logistic regression
        """
        if self.parameters == None:
            print("the parameters are not initialized yet")
            return
        
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        cache = {"Z1":Z1,
                 "A1":A1,
                 "Z2":Z2,
                 "A2":A2}
        self.cache = cache
        return A2
    
    def calculate_cost(self, A, Y):
        """
        calculate the cost
        Arguments:
        A -- the result after the logistic regression
        Y -- the "true" label of the training set

        Returns:
        cost -- the cost of the model now
        """
        m = Y.shape[1]
        logprobs = np.multiply(Y, np.log(A+0.0000001)) + np.multiply(1-Y, np.log(1-A+0.0000001))
        cost = -1/m * np.sum(logprobs)
        return cost

    def backward_propagation(self, A2, Y, X):
        """
        calculate the backward propagation
        Arguments:
        A2 -- the predicts of the logistic regression
        Y -- the labels of the training set
        X -- the inputs of the training set
        self.parameters -- python dictionary containing the params
        self.cache -- python dictionary containing the value through the logistic regression

        Return:
        grads -- python dictionary containing the gradient descent
        """

        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        Z2 = self.cache["Z2"]

        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = 1/m * np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1":dW1,
                 "db1":db1,
                 "dW2":dW2,
                 "db2":db2}
        return grads
        

    def update_params(self, grads, learning_rate=1.2):
        """
        update the params
        Arguments:
        grads -- python dictionary containing gradient descents
        self.parameters -- python dictionary containing the params
        learning_rate -- the pace of the gradient descent

        Return:
        Nothing
        """
        W1 = self.parameters["W1"] - learning_rate*grads["dW1"]
        b1 = self.parameters["b1"] - learning_rate*grads["db1"]
        W2 = self.parameters["W2"] - learning_rate*grads["dW2"]
        b2 = self.parameters["b2"] - learning_rate*grads["db2"]

        self.parameters = {"W1":W1,
                           "b1":b1,
                           "W2":W2,
                           "b2":b2}


    def nn_model(self, X, Y, n_h=4, iters=10000, print_cost=False):
        """
        this is the logistic regression model
        Arguments:
        X -- the input of the training data
        Y -- the label of the training data
        iters -- the loops of the training
        print_cost -- print the cost or not

        Returns:
        Nothing
        """
        n_x, n_y = self.get_shape(X, Y)
        self.initial_parameters(n_x=n_x, n_y=n_y)
        for i in range(0, iters):
            A2 = self.forward_propagation(X, Y)
            cost = self.calculate_cost(A2, Y)
            grads = self.backward_propagation(A2, Y, X)
            self.update_params(grads)
            if print_cost and i % 1000 == 0:
                print("cost after iter %i: %f" %(i, cost))


        


    def run(self):
        n_x, n_y = self.get_shape(x, y)
        self.initial_parameters(n_x=n_x, n_y=n_y)
        A2 = self.forward_propagation(x, y)
        cost = self.calculate_cost(A2, y)
        grads = self.backward_propagation(A2, y, x)
        self.update_params(grads)
        


if __name__ == "__main__":
    x, y = load_training_data()
    log_reg = logistic_regression()
    log_reg.nn_model(x, y, print_cost=True)
