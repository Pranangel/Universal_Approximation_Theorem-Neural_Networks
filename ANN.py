#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import numpy as np
from numpy import ndarray, matrix
import math

#https://www.geeksforgeeks.org/python/how-to-map-a-function-over-numpy-array/
def sigmoid(mat: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-1 * mat))

def softmax(mat: ndarray) -> ndarray:
    """Takes a matrix as an argument and calculates a probability distribution 
    represented by a matrix with the same shape."""

    sumExp = np.sum(np.exp(mat))
    return np.exp(mat) / sumExp

ACTIVATION_FUNCS = {
    "sigmoid" : sigmoid,
    "softmax" : softmax
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

#Layout: Start by pre-defining hidden layer sizes and activation functions used.
#TODO: HOW AM I SUPPOSED TO DO BACKPROP WITH OOP????
class BasicANN:
    """A basic class instantiating an artificial neural network."""
    def __init__(self) -> None:
        self.hiddenLayers = []
        self.__buildHiddenLayers()

    def __buildHiddenLayers(self) -> None:
        #input is (10,2)
        self.hiddenLayers.append(Layer(m=2, n=10, funcName="sigmoid"))
        self.hiddenLayers.append(Layer(m=10, n=10, funcName="sigmoid"))
        self.hiddenLayers.append(Layer(m=10, n=10, funcName="sigmoid"))
        self.hiddenLayers.append(Layer(m=10, n=1, funcName="softmax"))

    #TODO: fix stub
    def addLayer(): pass

    #TODO: implement loop
    def forwardPropagation(self, x: ndarray) -> ndarray:
        prevLayerOutput = self.hiddenLayers[0].forward(x)
        a2 = self.hiddenLayers[1].forward(prevLayerOutput)
        a3 = self.hiddenLayers[2].forward(a2)
        output = self.hiddenLayers[3].forward(a3)

        # for layer in self.hiddenLayers:
        #     output = layer.forward(prevLayerOutput)
        #     prevLayerOutput = output 

        return output

class Layer:
    
    def __init__(self, m: int, n: int, funcName: str):
        self.weights = np.random.rand(m, n)
        self.activationFunc = ACTIVATION_FUNCS[funcName] #TODO: what if string not valid?
        self.p = None #p for product of matrix multiplication
        self.a = None #a for activation function a(p)

    """Forward propagation, returns a Numpy array resulting from matrix mult w/ previous layer output (plus biases)"""
    def forward(self, input: ndarray) -> ndarray:
        self.p = np.matmul(input, self.weights)
        self.a = self.activationFunc(self.p)
        return self.a
    
    """Backward propagation to update weights."""
    def updateWeights(self, dw: ndarray, learnRate: int): #TODO: calc dw
        self.weights += dw * learnRate

#Forward prop
#Initializing input and weight matrices randomly.
n = 10
input = np.random.rand(n, 2)
a = BasicANN()
print(a.forwardPropagation(input))