#Author: Pranangel
#Purpose: This module contains wrapper classes housing activation functions and their methods.

import numpy as np
from numpy import ndarray
from Function import Function

#https://medium.com/analytics-vidhya/activation-functions-optimization-techniques-and-loss-functions-75a0eea0bc31
class ActivationFunction(Function):
    pass

class Sigmoid(ActivationFunction):
    @staticmethod
    def getFunc():
        return sigmoid
    
    @staticmethod
    def getDeriv():
        return derivSigmoid
    
def sigmoid(z: ndarray) -> ndarray:
    """Takes a matrix as an argument and applies the sigmoid function to every value in the matrix.
    
    Algebraically, sigmoid is defined as 1 / (1 + e^-z). However, this implementation uses
    (1 / 1 + exp(-z)) for z > 0 and exp(z) / (1 + exp(z)) for z < 0, avoiding overflow errors.

    This method creates two masks of the inputted matrix, one for values > 0 and another for values <= 0.
    The optimized version of the sigmoid is applied to a result matrix with respect to the masks.
    
    (Source: https://blog.dailydoseofds.com/p/sigmoid-and-softmax-are-not-implemented)
    """

    result = np.zeros_like(a=z, dtype=float)
    #Boolean matrices contain 0s (false) or 1s (true).
    positiveMask = z > 0  #Mask for values > 0
    negativeMask = z <= 0 #Mask for values <= 0

    #The masks are being used inside brackets to have the function apply only to values with true.
    a = np.exp(-1 * z[positiveMask])
    result[positiveMask] = 1 / (1 + a)

    a = np.exp(z[negativeMask])
    result[negativeMask] = a / (a + 1)
    
    return result

def derivSigmoid(z: ndarray) -> ndarray:
    a = sigmoid(z)
    return a * (1 - a)

class ReLU(ActivationFunction):
    @staticmethod
    def getFunc():
        return relu
    
    @staticmethod
    def getDeriv():
        return derivRelu
    
def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)

def derivRelu(x: ndarray) -> ndarray:
    positiveMask = x > 0
    result = np.zeros_like(a=x, shape=x.shape)
    result[positiveMask] = 1

    return result

#TODO
class GeLU:
    pass

#FIXME
# def softmax(mat: ndarray) -> ndarray:
#     """Takes a matrix as an argument and calculates a probability distribution 
#     represented by a matrix with the same shape."""

#     sumExp = np.sum(np.exp(mat))
#     return np.exp(mat) / sumExp
