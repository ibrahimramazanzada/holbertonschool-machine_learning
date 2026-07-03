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
