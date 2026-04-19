#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import numpy as np
from numpy import ndarray, matrix
import math

def sigmoid(z: ndarray) -> ndarray:
    """Takes a matrix as an argument and applies the sigmoid function to every value in the matrix.
    
    Algebraically, sigmoid is defined as 1 / (1 + e^-z). However, this implementation uses
    (1 / 1 + sigmoid(z)) for z > 0 and sigmoid(z) / (1 + sigmoid(z)) for z < 0, avoiding overflow errors.

    This method creates two masks of the inputted matrix, one for values > 0 and another for values <= 0.
    The optimized version of the sigmoid is applied to each mask, and then both values are added together
    and returned.
    
    (Source: https://blog.dailydoseofds.com/p/sigmoid-and-softmax-are-not-implemented)
    """

    positiveMatrix = np.multiply(z > 0, z)
    negativeMatrix = np.multiply(z < 0, z)

    positiveMatrix = np.exp(-1 * positiveMatrix)

    a = np.exp(negativeMatrix)
    negativeMatrix = a / (a + 1)
    
    return positiveMatrix + negativeMatrix

def derivSigmoid(z: ndarray) -> ndarray:
    a = sigmoid(z)
    return a * (1 - a)

def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)

def derivRelu(x: ndarray) -> ndarray:
    return np.ones_like(x) if x > 0 else np.zeros_like(x)

def softmax(mat: ndarray) -> ndarray:
    """Takes a matrix as an argument and calculates a probability distribution 
    represented by a matrix with the same shape."""

    sumExp = np.sum(np.exp(mat))
    return np.exp(mat) / sumExp

ACTIVATION_FUNCS = {
    "sigmoid" : sigmoid,
    "relu"    : relu,
    "softmax" : softmax
}

DERIV_ACTIVATION_FUNCS = {
    "sigmoid_d" : derivSigmoid,
    "relu_d"    : derivRelu,
}

class BasicANN:
    """BasicANN initializes a predefined artificial neural network. Architecture: WIP"""
    def __init__(self, input: ndarray) -> None:
        self.input = input

        self.hiddenLayers = []
        self.weights      = []
        self.biases       = []
        self.pOutputs     = []
        self.aOutputs     = []
        self.activations  = []

        self.__buildHiddenLayers()

    def __buildHiddenLayers(self) -> None:
        rows, columns = self.input.shape
        self.hiddenLayers.append(Layer(inputM=rows, inputN=columns, outputM=rows, outputN=rows, funcName="sigmoid"))
        h1M, h1N = self.hiddenLayers[0].getAOutputs().shape
        self.hiddenLayers.append(Layer(inputM=h1M, inputN=h1N, outputM=rows, outputN=1, funcName="sigmoid"))

    def forwardPropagation(self) -> ndarray:
        # i = 0
        a = self.input
        for layer in self.hiddenLayers:
            a = layer.forward(a)

            #Tracking every p, a, weight, and activation function
            # self.weights.append(layer.getWeights())
            # self.biases.append(layer.getBiases())
            # self.pOutputs.append(layer.getPOutputs())
            # self.aOutputs.append(a)
            # self.activations.append(layer.getActivationFunc())

        return a #return the output of the final layer
    
    #TODO:
    #   -enforce proper dimensionality of z argument
    #   -Add loop for partials
    def backPropagation(self, z: ndarray, learnRate: float):
        hl1 = self.hiddenLayers[0]
        hl2 = self.hiddenLayers[1]

        zPredicted = hl2.getAOutputs()

        w2 = hl2.getWeights()
        p2 = hl2.getPOutputs()
        derivActivation2 = hl2.getActivationDeriv()

        a1 = hl1.getAOutputs()
        p1 = hl1.getPOutputs()
        derivActivation1 = hl1.getActivationDeriv()

        #intermediate calculations
        #TODO:
            #dEdP2, dEdA1, dEdP1 combine np.matrix and ndarray with questionable transposes and broadcasting
        print(hl2.getActivationFunc()(p2))
        print(derivActivation2(p2))
        dEdP2 = np.matrix(2 * (zPredicted - z).flatten() * derivActivation2(p2).flatten()).T #TODO: add logic to handle activation function strings NOT in the dictionary
        dEdA1 = dEdP2 @ w2.T #one weight affects every single value in a row, hence matrix multiplication undoes it
        dEdP1 = dEdA1 * derivActivation1(p1)
        dP2dP1 = w2.T

        #Calculating partials and updating weights
        dEdW2 = (np.matrix(dEdP2).T @ a1).T
        dEdW1 = (dEdP1 @ self.input).T #the math doesn't feel right, but the shape is. (2, 10)

        self.hiddenLayers[0].updateParameters(dEdW1, 0, learnRate) #TODO: add code to handle bias updates
        self.hiddenLayers[1].updateParameters(dEdW2, 0, learnRate)
        # self.weights[len(self.weights) - 1] -= (learnRate * dEdW2)
        # self.weights[len(self.weights) - 2] -= (learnRate * dEdW1)

    def test(): pass
    
class Layer:
    """
    A Layer is a matrix with three properties: its dimensions n and m, and an activation function.
    Terminology: the p ('product') matrix is the product between the previous layer and weights, the a
    ('activation') matrix is the p matrix that has an activation function applied to it.
    
    A Layer should know the shape of the inputted matrix, the expected shape of the outputted matrix,
    and the activation function the programmer intends to use.
    """
    
    #TODO: handle checking for valid shapes
    """inputM and inputN are the dimensions of the inputted matrix, outputM and outputN are the dimensions of the outputted matrix."""
    def __init__(self, inputM: int, inputN: int, outputM: int, outputN: int, funcName: str):
        self.activationFunc  = ACTIVATION_FUNCS[funcName] #TODO: what if string is invalid?
        self.activationDeriv = DERIV_ACTIVATION_FUNCS[f"{funcName}_d"] #TODO: what if string is invalid?
        
        self.weights = np.random.rand(inputN, outputN)
        self.biases = np.random.rand(outputM, outputN)
        
        self.p = np.random.rand(outputM, outputN)
        self.a = np.random.rand(outputM, outputN)

    """Forward propagation algorithm: returns a numpy array of matrix multiplication and an applied activation
    function."""
    #Optimization: Fix: implement mini-batching or process per-sample; ensure shapes reflect (batch_size, features).
    def forward(self, input: ndarray) -> ndarray:
        #matrix multiplication
        self.p = input @ self.weights
        self.a = self.activationFunc(self.p)
        return self.a
    
    """Backward propagation to update weights and biases."""
    def updateParameters(self, dw: ndarray, db: ndarray, learnRate: int):
        self.weights -= dw * learnRate
        self.biases -= db * learnRate

    def getActivationFunc(self): #Returns a function wrapper
        return self.activationFunc
    
    def getActivationDeriv(self): #Returns a function wrapper for the derivative of the activation function
        return self.activationDeriv

    def getWeights(self) -> ndarray:
        return self.weights
    
    def getBiases(self) -> ndarray:
        return self.biases
    
    def getPOutputs(self) -> ndarray:
        return self.p
    
    def getAOutputs(self) -> ndarray:
        return self.a

#Loading data from csv and loading into a numpy matrix
import pandas as pd
dataDF = pd.read_csv("training_data.csv")
dataDF = dataDF.sample(frac=1).reset_index(drop=True)

trainInputs = dataDF[["x", "y"][:8000]].to_numpy()
trainOutputs = dataDF[["z"][:8000]].to_numpy()

test = dataDF[["x", "y"][8000:10000]].to_numpy()

a = BasicANN(trainInputs)
epochs = 1
out = None
for i in range(epochs):
    print(f"Epoch: {i + 1}")
    out = a.forwardPropagation()
    a.backPropagation(trainOutputs, 0.1)
    print(f"Inputs: {trainInputs}")
    print(f"Predicted: {out}")
    print(f"Actual: {trainOutputs}")

"""TEST 1 (No learning occurring, need to pinpoint bugs and optimizations):
Epoch: 1
Inputs: [[-2.50320799 -1.10966418]
 [-0.56200661  2.59808924]
 [ 2.95085971 -0.61225634]
 ...
 [-2.95008035 -1.02161336]
 [ 1.81982744 -2.92119976]
 [ 2.39158146  2.68269902]]
Predicted: [[1.]
 [1.]
 [1.]
 ...
 [1.]
 [1.]
 [1.]]
Actual: [[0.09195113]
 [0.10548865]
 [0.05551914]
 ...
 [0.04493848]
 [0.02304197]
 [0.01638382]]
Epoch: 2
Inputs: [[-2.50320799 -1.10966418]
 [-0.56200661  2.59808924]
 [ 2.95085971 -0.61225634]
 ...
 [-2.95008035 -1.02161336]
 [ 1.81982744 -2.92119976]
 [ 2.39158146  2.68269902]]
Predicted: [[1.]
 [1.]
 [1.]
 ...
 [1.]
 [1.]
 [1.]]
Actual: [[0.09195113]
 [0.10548865]
 [0.05551914]
 ...
 [0.04493848]
 [0.02304197]
 [0.01638382]]
Epoch: 3
Inputs: [[-2.50320799 -1.10966418]
 [-0.56200661  2.59808924]
 [ 2.95085971 -0.61225634]
 ...
 [-2.95008035 -1.02161336]
 [ 1.81982744 -2.92119976]
 [ 2.39158146  2.68269902]]
Predicted: [[1.]
 [1.]
 [1.]
 ...
 [1.]
 [1.]
 [1.]]
Actual: [[0.09195113]
 [0.10548865]
 [0.05551914]
 ...
 [0.04493848]
 [0.02304197]
 [0.01638382]]
"""

""" TEST 2 (with activation function performance issues fixed, but still no learning occurring)
Epoch: 1
Inputs: [[-1.73834293  1.40594045]
 [-1.88318461 -1.76845482]
 [ 1.35340949 -0.339626  ]
 ...
 [ 0.73734285 -2.72708075]
 [-2.74829266 -0.96171274]
 [-1.73682907  1.57032535]]
Predicted: [[0.5]
 [0.5]
 [0.5]
 ...
 [0.5]
 [0.5]
 [0.5]]
Actual: [[0.20370682]
 [0.1195116 ]
 [0.53806849]
 ...
 [0.07884134]
 [0.06729748]
 [0.17462177]]
Epoch: 2
Inputs: [[-1.73834293  1.40594045]
 [-1.88318461 -1.76845482]
 [ 1.35340949 -0.339626  ]
 ...
 [ 0.73734285 -2.72708075]
 [-2.74829266 -0.96171274]
 [-1.73682907  1.57032535]]
Predicted: [[0.5]
 [0.5]
 [0.5]
 ...
 [0.5]
 [0.5]
 [0.5]]
Actual: [[0.20370682]
 [0.1195116 ]
 [0.53806849]
 ...
 [0.07884134]
 [0.06729748]
 [0.17462177]]
Epoch: 3
Inputs: [[-1.73834293  1.40594045]
 [-1.88318461 -1.76845482]
 [ 1.35340949 -0.339626  ]
 ...
 [ 0.73734285 -2.72708075]
 [-2.74829266 -0.96171274]
 [-1.73682907  1.57032535]]
Predicted: [[0.5]
 [0.5]
 [0.5]
 ...
 [0.5]
 [0.5]
 [0.5]]
Actual: [[0.20370682]
 [0.1195116 ]
 [0.53806849]
 ...
 [0.07884134]
 [0.06729748]
 [0.17462177]]
Epoch: 4
Inputs: [[-1.73834293  1.40594045]
 [-1.88318461 -1.76845482]
 [ 1.35340949 -0.339626  ]
 ...
 [ 0.73734285 -2.72708075]
 [-2.74829266 -0.96171274]
 [-1.73682907  1.57032535]]
Predicted: [[0.5]
 [0.5]
 [0.5]
 ...
 [0.5]
 [0.5]
 [0.5]]
Actual: [[0.20370682]
 [0.1195116 ]
 [0.53806849]
 ...
 [0.07884134]
 [0.06729748]
 [0.17462177]]
Epoch: 5
Inputs: [[-1.73834293  1.40594045]
 [-1.88318461 -1.76845482]
 [ 1.35340949 -0.339626  ]
 ...
 [ 0.73734285 -2.72708075]
 [-2.74829266 -0.96171274]
 [-1.73682907  1.57032535]]
Predicted: [[0.5]
 [0.5]
 [0.5]
 ...
 [0.5]
 [0.5]
 [0.5]]
Actual: [[0.20370682]
 [0.1195116 ]
 [0.53806849]
 ...
 [0.07884134]
 [0.06729748]
 [0.17462177]]
"""
