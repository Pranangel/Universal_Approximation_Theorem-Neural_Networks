#Author: Pranangel
#Purpose: This module contains wrapper classes housing error functions and their methods.

import numpy as np
from numpy import ndarray
from abstract_function import Function

class ErrorFunction(Function):
    """
    Wrapper class for all losses. Currently supports:

    
        -SqaredError
        -MSE
            -MeanSquaredError (scalar)
            -PerSampleMSE
            -PerOutputMSE
        -MeanAbsoluteError
    """
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

#TODO: add per sample (row-wise), per output (column/neuron-wise)
def meanSquaredError(predicted: ndarray, actual: ndarray): #TODO: outputs floating[Any]
    return np.mean((predicted - actual) ** 2)
    
def derivMeanSquaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return 2 * (predicted - actual) / predicted.size

class PerOutputMSE(ErrorFunction):
    """
    Column-wise mean squared error. This class expects that matrix rows represent
    samples and columns represent neurons; returns one MSE value per output.
    """

    @staticmethod
    def getFunc():
        return perOutputMSE

    @staticmethod
    def getDeriv():
        return derivPerOutputMSE

def perOutputMSE(predicted: ndarray, actual: ndarray): #TODO: outputs floating[Any]
    return np.mean((predicted - actual) ** 2, axis=0)
    
def derivPerOutputMSE(predicted: ndarray, actual: ndarray) -> ndarray:
    samples, neurons = actual.shape
    return 2 * (predicted - actual) / samples

class PerSampleMSE(ErrorFunction):
    """
    Row-wise mean squared error. This class expects that matrix rows represent
    samples and columns represent neurons; returns one MSE value per sample.
    """

    @staticmethod
    def getFunc():
        return perSampleMSE

    @staticmethod
    def getDeriv():
        return derivPerSampleMSE

def perSampleMSE(predicted: ndarray, actual: ndarray): #TODO: outputs floating[Any]
    return np.mean((predicted - actual) ** 2, axis=1)
    
def derivPerSampleMSE(predicted: ndarray, actual: ndarray) -> ndarray:
    samples, neurons = actual.shape
    return 2 * (predicted - actual) / neurons

class MeanAbsoluteError(ErrorFunction):

    @staticmethod
    def getFunc():
        return meanAbsError

    @staticmethod
    def getDeriv():
        return derivMeanAbsError

def meanAbsError(predicted: ndarray, actual: ndarray):
    return np.mean(np.abs(predicted - actual))

def derivMeanAbsError(predicted: ndarray, actual: ndarray):
    return np.sign(predicted - actual) / predicted.size

#TODO
# class BinaryCrossEntropy(ErrorFunction):
#     pass
