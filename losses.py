#Author: Pranangel
#Purpose: This module contains a wrapper class housing loss functions.

import numpy as np
from numpy import ndarray

class LossFunction():
    """
    Wrapper class for all losses. Currently supports:

    
        -SqaredError
        -MSE
            -MeanSquaredError (scalar)
            -PerSampleMSE
            -PerOutputMSE
        -MeanAbsoluteError
    """

class SquaredError(LossFunction):
    @staticmethod
    def getLoss(predicted: ndarray, actual: ndarray):
        return SquaredError.__squaredError(predicted, actual)

    @staticmethod
    def getDeriv(predicted: ndarray, actual: ndarray):
        return SquaredError.__deriv(predicted, actual)

    @staticmethod
    def __squaredError(predicted: ndarray, actual: ndarray) -> ndarray:
        return (predicted - actual) ** 2

    @staticmethod
    def __deriv(predicted: ndarray, actual: ndarray) -> ndarray:
        return 2 * (predicted - actual)

class MeanSquaredError(LossFunction):
    @staticmethod
    def getLoss(predicted: ndarray, actual: ndarray):
        return MeanSquaredError.__mse(predicted, actual)

    @staticmethod
    def getDeriv(predicted: ndarray, actual: ndarray):
        return MeanSquaredError.__deriv(predicted, actual)

    @staticmethod
    def __mse(predicted: ndarray, actual: ndarray): #Outputs floating[Any]
        return np.mean((predicted - actual) ** 2)
        
    @staticmethod
    def __deriv(predicted: ndarray, actual: ndarray) -> ndarray:
        return 2 * (predicted - actual) / predicted.size

class PerOutputMSE(LossFunction):
    """
    Column-wise mean squared error. This class expects that matrix rows represent
    samples and columns represent neurons; returns one MSE value per output.
    """

    @staticmethod
    def getLoss(predicted: ndarray, actual: ndarray):
        return PerOutputMSE.__perOutputMSE(predicted, actual)

    @staticmethod
    def getDeriv(predicted: ndarray, actual: ndarray):
        return PerOutputMSE.__deriv(predicted, actual)

    @staticmethod
    def __perOutputMSE(predicted: ndarray, actual: ndarray): #TODO: outputs floating[Any]
        return np.mean((predicted - actual) ** 2, axis=0)
        
    @staticmethod
    def __deriv(predicted: ndarray, actual: ndarray) -> ndarray:
        samples, neurons = actual.shape
        return 2 * (predicted - actual) / samples

class PerSampleMSE(LossFunction):
    """
    Row-wise mean squared error. This class expects that matrix rows represent
    samples and columns represent neurons; returns one MSE value per sample.
    """

    @staticmethod
    def getLoss(predicted: ndarray, actual: ndarray):
        return PerSampleMSE.__perSampleMSE(predicted, actual)

    @staticmethod
    def getDeriv(predicted: ndarray, actual: ndarray):
        return PerSampleMSE.__deriv(predicted, actual)

    @staticmethod
    def __perSampleMSE(predicted: ndarray, actual: ndarray): #TODO: outputs floating[Any]
        return np.mean((predicted - actual) ** 2, axis=1)
        
    @staticmethod
    def __deriv(predicted: ndarray, actual: ndarray) -> ndarray:
        samples, neurons = actual.shape
        return 2 * (predicted - actual) / neurons

class MeanAbsoluteError(LossFunction):

    @staticmethod
    def getLoss(predicted: ndarray, actual: ndarray):
        return MeanAbsoluteError.__meanAbsError(predicted, actual)

    @staticmethod
    def getDeriv(predicted: ndarray, actual: ndarray):
        return MeanAbsoluteError.__deriv(predicted, actual)

    @staticmethod
    def __meanAbsError(predicted: ndarray, actual: ndarray):
        return np.mean(np.abs(predicted - actual))

    @staticmethod
    def __deriv(predicted: ndarray, actual: ndarray):
        return np.sign(predicted - actual) / predicted.size

#TODO
# class BinaryCrossEntropy(ErrorFunction):
#     pass
LOSSES = {
    "squared_error":  SquaredError(),
    "mse":            MeanSquaredError(),
    "mse_per_sample": PerSampleMSE(),
    "mse_per_output": PerOutputMSE(),
    "mae":            MeanAbsoluteError(),
    "":               MeanSquaredError(),
    None:             MeanSquaredError()
}
