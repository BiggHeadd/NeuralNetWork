# !/bin/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-12-1

import numpy as np
from util import load_training_data

class logistic_regression_Ng:

    def __init__(self):
        pass

    def layer_sizes(self, X, Y):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden lyaer
        n_y -- the size of the output layer
        """

        print(X.shape)
        print(Y.shape)
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[1]
        return n_x, n_h, n_y
    
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                    w1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vetor of shape (n_h, 1)
                    w2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vetor of shape (n_y, 1)
        """

        w1 = np.random.randn(n_h, n_x)
        b1 = np.random.randn(n_h, 1)
        w2 = np.random.randn(n_y, n_h)
        b2 = np.random.randn(n_y, 1)

        assert (w1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (w2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        params = dict()
        params['w1'] = w1
        params['b1'] = b1
        params['w2'] = w2
        params['b2'] = b2
        return params

    def sigmoid(self, z):
        """
        Arguments:
        z -- input value of the data (np.array)

        returns:
        sigmoid -- the value of calculating the sigmoid(z)
        """
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid


    def forward_propagation(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialize_parameters)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z1" and "A2"
        """
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']

        Z1 = np.dot(w1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(w2, A1) + b2
        A2 = self.sigmoid(Z2)
        cache = {'Z1': Z1,
                 'A1': A1,
                 'Z2': Z2,
                 'A2': A2}

        return A2, cache

    def compute_cost(self, A2, Y, parameters):
        """
        Computes the cross-entropy cost given in equation

        Arguments:
        A2 -- the sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vetors of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2, b2
        
        Returns:
        cost -- cross-entropy cost given equation
        """

        m = Y.shape[0]
        print(m)
        logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
        cost = -1/m * np.sum(logprobs)
        cost = np.squeeze(cost)
        assert(isinstance(cost, float))
        return cost

    def run(self):
        n_x, n_h, n_y = self.layer_sizes(x, y)
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        A2, cache = self.forward_propagation(x, parameters)
        cost = self.compute_cost(A2, y, parameters)
        print(cost)

if __name__ == "__main__":
    x, y = load_training_data()
    clf = logistic_regression_Ng()
    clf.run()
