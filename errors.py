#Author: Pranangel
#Purpose: This module contains wrapper classes housing error functions and their methods.

import numpy as np
from numpy import ndarray
from Function import Function

class ErrorFunction(Function):
    pass

class SquaredError(ErrorFunction):
    @staticmethod
    def getFunc():
        return squaredError

    @staticmethod
    def getDeriv():
        return derivSquaredError

def squaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return (predicted - actual) ** 2

def derivSquaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return 2 * (predicted - actual)

class MeanSquaredError(ErrorFunction):
    @staticmethod
    def getFunc():
        return meanSquaredError

    @staticmethod
    def getDeriv():
        return derivMeanSquaredError

#TODO: add per sample (axis=1), per output (axis=0)
def meanSquaredError(predicted: ndarray, actual: ndarray): #TODO: outputs floating[Any]
    return np.mean((predicted - actual) ** 2)
    
def derivMeanSquaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return 2 * (predicted - actual) / predicted.size

#TODO
# class BinaryCrossEntropy(ErrorFunction):
#     pass

#FIXME
# class MeanAbsoluteError(ErrorFunction):
    # @staticmethod
    # def meanAbsError(predicted: ndarray, actual: ndarray) -> ndarray:
    #     m1, n1 = predicted.shape #TODO: check for same-size shape
    #     m2, n2 = actual.shape
    #     return np.abs(predicted - actual)  / n1

    # @staticmethod
    # def derivMeanAbsError(predicted: ndarray, actual: ndarray) -> ndarray:
    #     m1, n1 = predicted.shape #TODO: check for same-size shape
    #     m2, n2 = actual.shape

    #     diff = predicted - actual
    #     copy = np.ones((m1, m2))

    #     if (diff > 0):
    #         return copy * (1 / n1)
    #     elif (diff < 0):
    #         return copy * (-1 / n1)
    #     return copy * 0
