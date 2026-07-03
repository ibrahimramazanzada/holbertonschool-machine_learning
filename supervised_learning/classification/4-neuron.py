#!/usr/bin/env python3
'''Beginning of neural network implementation'''
import numpy as np


class Neuron():
    '''Neuron class'''
    def __init__(self, nx):
        '''Initializes the neuron'''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Returns the weights vector'''
        return self.__W

    @property
    def b(self):
        '''Returns the bias'''
        return self.__b

    @property
    def A(self):
        '''Returns the activated output'''
        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''Calculates cost of model using logistic regression'''
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions'''
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
