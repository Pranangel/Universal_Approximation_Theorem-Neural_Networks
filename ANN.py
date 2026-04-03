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

# #Forward prop
# #Initializing input and weight matrices randomly.
# n = 10
# input = np.random.rand(n, 2)
# w1 = np.random.rand(2, n)
# w2 = np.random.rand(n, 1)

# #First hidden layer
# p1 = np.matmul(input, w1)
# a1 = sigmoid(p1)

# #Softmax output layer
# p2 = np.matmul(a1, w2)
# a2 = softmax(p2)
# z = a2

class BasicANN:
    """BasicANN initializes a predefined artificial neural network. Architecture: WIP"""
    def __init__(self) -> None:
        self.hiddenLayers = []
        self.__buildHiddenLayers()

        self.input = ndarray
        self.weights     = []
        self.pOutputs    = []
        self.aOutputs    = []
        self.activations = []

    def __buildHiddenLayers(self) -> None:
        #input is (10,2)
        self.hiddenLayers.append(Layer(m=2, n=10, funcName="sigmoid"))

    #TODO: implement loop
    def forwardPropagation(self, x: ndarray) -> ndarray:
        self.input = x

        layer = self.hiddenLayers[0]
        a1 = layer.forward(x)

        #Tracking every p, a, weight, and activation function
        self.weights.append(layer.getWeights())
        self.pOutputs.append(layer.getPOutputs())
        self.aOutputs.append(a1)
        self.activations.append(layer.getActivationFunc())

        # for layer in self.hiddenLayers:
        #     output = layer.forward(prevLayerOutput)
        #     prevLayerOutput = output 

        return a1
    
    def backPropagation(self, z: ndarray, learnRate: float): #TODO: enforce proper dimensionality of z argument
        zPredicted = self.aOutputs[len(self.aOutputs) - 1]
        prevP = self.pOutputs[len(self.aOutputs) - 1]
        prevActivation = self.activations[len(self.aOutputs) - 1]
        dPrevActivation = self.__getActivationDeriv(prevActivation)

        dEdA = 2 * (zPredicted - z)
        dAdP = DERIV_ACTIVATION_FUNCS[dPrevActivation](prevP) if dPrevActivation != "" else None
        dPdW1 = self.input

        self.weights[len(self.weights) - 1] -= (learnRate * dEdA * dAdP * dPdW1)

    def __getActivationDeriv(self, func) -> str: #func is a parameter of type function
        if (func.__name__ in ACTIVATION_FUNCS.keys()):
            return f"{func.__name__}_d"
        return ""

class CustomANN:
    """A basic class instantiating an artificial neural network."""
    def __init__(self) -> None:
        self.hiddenLayers = []
        self.__buildHiddenLayers()

        self.pOutputs = []
        self.aOutputs = []
        self.weights = []

    def __buildHiddenLayers(self) -> None:
        #input is (10,2)
        self.hiddenLayers.append(Layer(m=2, n=10, funcName="sigmoid"))

    #TODO: fix stub
    def addLayer(): pass

    #TODO: implement loop
    def forwardPropagation(self, x: ndarray) -> ndarray:
        a1 = self.hiddenLayers[0].forward(x) #a1 is the output
        self.pOutputs.append(self.hiddenLayers[0].getPOutputs())
        self.aOutputs.append(a1)
        self.weights.append(self.hiddenLayers[0].getWeights())

        # for layer in self.hiddenLayers:
        #     output = layer.forward(prevLayerOutput)
        #     prevLayerOutput = output 

        return a1
    
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
        self.p = np.matmul(input, self.weights)
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
a = BasicANN()
print(a.forwardPropagation(input))
a.backPropagation(np.random.rand(10, 1), 0.1)
