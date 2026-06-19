#Types of ReLU: https://apxml.com/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-variants
#GeLU specifically: https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c
#https://medium.com/analytics-vidhya/activation-functions-optimization-techniques-and-loss-functions-75a0eea0bc31

#Author: Pranangel
#Purpose: This module contains wrapper classes housing activation functions and their methods.

import numpy as np
from numpy import ndarray

class ActivationFunction():
    """
    Wrapper class for all activations. Currently supports:

    
        -Sigmoid
        -ReLU variations
            -Normal
            -Leaky
        -Identity
    """

class Sigmoid(ActivationFunction):

    @staticmethod
    def getFunc(input: ndarray):
        return Sigmoid.__func(input)
    
    @staticmethod
    def getDeriv(input: ndarray):
        return Sigmoid.__deriv(input)
    

    @staticmethod
    def __func(x: ndarray) -> ndarray:
        """Takes a matrix as an argument and applies the sigmoid function to every value in the matrix.
        
        Algebraically, sigmoid is defined as 1 / (1 + exp(-z)). However, this implementation uses
        (1 / 1 + exp(-z)) for z > 0 and exp(z) / (1 + exp(z)) for z < 0, avoiding overflow errors.

        This method creates two masks of the inputted matrix, one for values > 0 and another for values <= 0.
        The optimized version of the sigmoid is applied to a result matrix with respect to the masks.
        
        (Source: https://blog.dailydoseofds.com/p/sigmoid-and-softmax-are-not-implemented)
        """

        result = np.zeros_like(a=x, dtype=float)
        #Boolean matrices contain 0s (false) or 1s (true).
        positiveMask = x > 0  #Mask for values > 0
        negativeMask = x <= 0 #Mask for values <= 0

        #The masks are being used inside brackets to have the function apply only to values with true.
        a = np.exp(-1 * x[positiveMask])
        result[positiveMask] = 1 / (a + 1)

        a = np.exp(x[negativeMask])
        result[negativeMask] = a / (a + 1)
        
        return result

    @staticmethod
    def __deriv(x: ndarray) -> ndarray:
        a = Sigmoid.__func(x)
        return a * (1 - a)

class ReLU(ActivationFunction):
    @staticmethod
    def getFunc(input: ndarray):
        return ReLU.__func(input)
    
    @staticmethod
    def getDeriv(input: ndarray):
        return ReLU.__deriv(input)
    
    @staticmethod
    def __func(x: ndarray) -> ndarray:
        return np.maximum(0, x)

    @staticmethod
    def __deriv(x: ndarray) -> ndarray:
        positiveMask = x > 0
        result = np.zeros_like(a=x, shape=x.shape)
        result[positiveMask] = 1

        return result

class LeakyReLU(ReLU):
    """Note:: Uses a coefficient of 0.003 for negatives. Also extends ReLU."""
    scalar = 0.003

    @staticmethod
    def getFunc(input: ndarray):
        return LeakyReLU.__func(input)
    
    @staticmethod
    def getDeriv(input: ndarray):
        return LeakyReLU.__deriv(input)
    
    @staticmethod
    def __func(x: ndarray) -> ndarray:
        negativeMask = x < 0
        result = np.ones_like(a=x, shape=x.shape)

        result *= x
        result[negativeMask] *= LeakyReLU.scalar

        return result

    @staticmethod
    def __deriv(x: ndarray) -> ndarray:
        negativeMask = x < 0
        result = np.ones_like(a=x, shape=x.shape)

        result[negativeMask] *= LeakyReLU.scalar

        return result

#TODO
# class GeLU:
#     pass

class Identity(ActivationFunction):

    @staticmethod
    def getFunc(input: ndarray):
        return Identity.__func(input)

    @staticmethod
    def getDeriv(input: ndarray):
        return Identity.__deriv(input)

    @staticmethod
    def __func(x: ndarray) -> ndarray:
        return x

    @staticmethod
    def __deriv(x: ndarray) -> ndarray:
        return np.ones(x.shape)

#TODO
# class Softmax(ActivationFunction):
#     pass

ACTIVATIONS = {
    "sigmoid": Sigmoid(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "identity": Identity(),
    "": Identity(),
    None: Identity()
}
