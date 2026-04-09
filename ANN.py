#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import numpy as np
from numpy import ndarray, matrix
import math
# import pyfunc

#https://www.geeksforgeeks.org/python/how-to-map-a-function-over-numpy-array/
def sigmoid(mat: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-1 * mat))

def derivSigmoid(mat: ndarray) -> ndarray:
    return np.exp(-1 * mat) / (1 + np.exp(-1 * mat))

def relu(x: int) -> int:
    return x if x > 0 else 0

def derivRelu(x: int) -> int:
    return 1 if x > 0 else 0

def softmax(mat: ndarray) -> ndarray:
    """Takes a matrix as an argument and calculates a probability distribution 
    represented by a matrix with the same shape."""

    sumExp = np.sum(np.exp(mat))
    return np.exp(mat) / sumExp

ACTIVATION_FUNCS = {
    "sigmoid" : sigmoid,
    "relu"    : np.vectorize(relu),
    "softmax" : softmax
}

DERIV_ACTIVATION_FUNCS = {
    "sigmoid_d" : sigmoid,
    "relu_d"    : np.vectorize(derivRelu),
}

class BasicANN:
    """BasicANN initializes a predefined artificial neural network. Architecture: WIP"""
    def __init__(self, input: ndarray) -> None:
        self.input = input

        self.hiddenLayers = []
        self.weights      = []
        self.pOutputs     = []
        self.aOutputs     = []
        self.activations  = []

        self.__buildHiddenLayers()

    def __buildHiddenLayers(self) -> None:
        #input is (10,2)
        rows, columns = self.input.shape
        self.hiddenLayers.append(Layer(m=2, n=rows, funcName="sigmoid"))
        self.hiddenLayers.append(Layer(m=rows, n=1, funcName="sigmoid"))

    def forwardPropagation(self) -> ndarray:
        i = 0
        a = self.input
        for layer in self.hiddenLayers:
            a = layer.forward(a)

            #Tracking every p, a, weight, and activation function
            self.weights.append(layer.getWeights())
            self.pOutputs.append(layer.getPOutputs())
            self.aOutputs.append(a)
            self.activations.append(layer.getActivationFunc())

        return self.aOutputs[len(self.aOutputs) - 1] #return the output of the final layer
    
    #TODO:
    #   -enforce proper dimensionality of z argument
    #   -Add loop for partials
    def backPropagation(self, z: ndarray, learnRate: float):
        zPredicted = self.aOutputs[len(self.aOutputs) - 1]
        p1 = self.pOutputs[len(self.aOutputs) - 2]
        p2 = self.pOutputs[len(self.aOutputs) - 1]
        prevWeight = self.weights[len(self.weights) - 1]
        prevActivation = self.activations[len(self.aOutputs) - 1]
        dPrevActivation = self.__getActivationDeriv(prevActivation)

        #intermediate calculations
        dEdP2 = np.matrix(2 * (zPredicted - z).flatten() * DERIV_ACTIVATION_FUNCS[dPrevActivation](p2).flatten()).T #TODO: add logic to handle activation function strings NOT in the dictionary
        dEdA1 = dEdP2 @ prevWeight.T #one weight affects every single value in a row, hence matrix multiplication undoes it
        dEdP1 = dEdA1 * DERIV_ACTIVATION_FUNCS[dPrevActivation](p1)

        #Calculating partials and updating weights
        dEdW2 = (np.matrix(dEdP2).T @ self.aOutputs[0]).T #self.aOutputs[0] is a1
        dEdW1 = (dEdP1 @ self.input).T #the math doesn't feel right, but the shape is. (2, 10)

        self.weights[len(self.weights) - 1] -= (learnRate * dEdW2)
        self.weights[len(self.weights) - 2] -= (learnRate * dEdW1)

    def test(): pass

    def __getActivationDeriv(self, func) -> str: #func is a parameter of type function
        if (func.__name__ in ACTIVATION_FUNCS.keys()):
            return f"{func.__name__}_d"
        return ""
    
class Layer:
    """A Layer is a matrix with three properties: its dimensions n and m, and an activation function.
    Terminology: the p ('product') matrix is the product between the previous layer and weights, the a
    ('activation') matrix is the p matrix that has an activation function applied to it."""
    
    def __init__(self, m: int, n: int, funcName: str):
        self.activationFunc = ACTIVATION_FUNCS[funcName] #TODO: what if string is invalid?
        self.weights = np.random.rand(m, n)
        self.p = np.random.rand(n, m)
        self.a = np.random.rand(n, m)

    """Forward propagation algorithm: returns a numpy array of matrix multiplication and an applied activation
    function."""
    def forward(self, input: ndarray) -> ndarray:
        #matrix multiplication
        self.p = input @ self.weights
        self.a = self.activationFunc(self.p)
        return self.a
    
    """Backward propagation to update weights."""
    def updateWeights(self, dw: ndarray, learnRate: int): #TODO: calc dw
        self.weights += dw * learnRate

    def getActivationFunc(self): #returns a function
        return self.activationFunc

    def getWeights(self) -> ndarray:
        return self.weights
    
    def getPOutputs(self) -> ndarray:
        return self.p
    
    def getAOutputs(self) -> ndarray:
        return self.a

#Forward prop
#Initializing input and weight matrices randomly.
n = 10
input = np.random.rand(n, 2)
a = BasicANN(input)
a.forwardPropagation()
a.backPropagation(np.random.rand(10, 1), 0.1)
