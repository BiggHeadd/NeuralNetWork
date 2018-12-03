# !/bin/.pyenv/versions/3.6.6/python3
# -*- coding: utf-8 -*-
# Edited by bighead 18-12-1

import numpy as np
from util import load_training_data, load_test_data

class logistic_regression_Ng:

    parameters = None

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
        n_y = Y.shape[0]
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

        m = Y.shape[1]
        logprobs = np.multiply(Y, np.log(A2+0.0000001)) + np.multiply(1-Y, np.log(1-A2+0.0000001))
        cost = -1/m * np.sum(logprobs)
        cost = np.squeeze(cost)
        assert(isinstance(cost, float))
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, numberof examples)

        Returns:
        grads -- python dictionary contraining your gradients with respect to different parameters
        """
        m = X.shape[1]
        
        Z1 = cache['Z1']
        A1 = cache['A1']
        Z2 = cache['Z2']
        A2 = cache['A2']

        W2 = parameters['w2']
        b2 = parameters['b2']
        W1 = parameters['w1']
        b1 = parameters['b1']

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

    def update_parameters(self, parameters, grads, learning_rate=1.2):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        W1 = parameters['w1']
        b1 = parameters['b1']
        W2 = parameters['w2']
        b2 = parameters['b2']

        dW1 = grads['dW1']
        dW2 = grads['dW2']
        db1 = grads['db1']
        db2 = grads['db2']

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"w1":W1,
                      "w2":W2,
                      "b1":b1,
                      "b2":b2}
        
        return parameters

    def nn_model(self, X, Y, n_h, num_iterations = 10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        n_x, nothing, n_y = self.layer_sizes(x, y)
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        W1 = parameters['w1']
        b1 = parameters['b1']
        W2 = parameters['w2']
        b2 = parameters['b2']
        for i in range(0, num_iterations):
            A2, cache = self.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, Y, parameters)
            if i % 1000 == 0 and print_cost:
                print("Cost after iteration %i: %f" %(i, cost))
            grads = self.backward_propagation(parameters, cache, X, Y)
            parameters = self.update_parameters(parameters, grads)
        self.parameters = parameters
        return parameters

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each examples in X

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)

        Returns:
        predictions -- vector of predictions of our model (0/1)
        """
        A2, cache = self.forward_propagation(X, self.parameters)
        threshold = 0.5
        predictions = A2 > threshold

        return predictions



    def run(self):
        n_x, n_h, n_y = self.layer_sizes(x, y)
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        A2, cache = self.forward_propagation(x, parameters)
        cost = self.compute_cost(A2, y, parameters)
        print(cost)
        grads = self.backward_propagation(parameters, cache, x, y)
        self.update_parameters(parameters, grads)


if __name__ == "__main__":
    x, y = load_training_data()
    clf = logistic_regression_Ng()
    clf.nn_model(X=x, Y=y, n_h=4, print_cost=True)
    x_test = load_test_data()
    y_predict = clf.predict(x_test)
    print(y_predict)
    print(np.mean(y_predict))
