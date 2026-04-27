#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import math
import numpy as np
from numpy import ndarray, matrix

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

        #TODO: store in dictionary?
        self.hiddenLayers = []
        self.weights      = []
        self.biases       = []
        self.pOutputs     = []
        self.aOutputs     = []
        self.activations  = []

        self.__buildHiddenLayers()

    #Private method for automatic initialization
    def __buildHiddenLayers(self) -> None:
        rows, columns = self.input.shape
        hl1Neurons = 5
        self.hiddenLayers.append(Layer(inputM=rows, inputN=columns, neurons=hl1Neurons, funcName="sigmoid"))
        
        h1M, h1N = self.hiddenLayers[0].getAOutputs().shape
        hl2Neurons = 1
        self.hiddenLayers.append(Layer(inputM=h1M, inputN=h1N, neurons=hl2Neurons, funcName="sigmoid"))

    def forwardPropagation(self) -> ndarray:
        i = 0
        a = self.input
        for i in range(len(self.hiddenLayers)):
            layer = self.hiddenLayers[i]
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

        # print("Backpropagation:\n")

        zPredicted       = hl2.getAOutputs()
        w2               = hl2.getWeights()
        p2               = hl2.getPOutputs()
        derivActivation2 = hl2.getActivationDeriv()
        a1               = hl1.getAOutputs()
        p1               = hl1.getPOutputs()
        derivActivation1 = hl1.getActivationDeriv()


        #intermediate calculations
        #TODO: derivActivation2(p2) is producing numbers close to zero because sigmoid??
        dEdP2 = np.multiply(2 * (zPredicted - z), derivActivation2(p2))
        dEdA1 = dEdP2 @ w2.T #Matrix multiplication undoes what the weights did to produce p2
        dEdP1 = np.multiply(dEdA1, derivActivation1(p1))

        #Calculating partials and updating weights
        dEdW2 = a1.T @ dEdP2 #shape(n,n) * (n,1) = (n,1)
        dEdW1 = self.input.T @ dEdP1
        dEdB2 = dEdP2
        dEdB1 = dEdP1

        self.hiddenLayers[0].updateParameters(dEdW1, dEdB1, learnRate)
        self.hiddenLayers[1].updateParameters(dEdW2, dEdB2, learnRate)

    def test(): pass
    
class Layer:
    index = 0 #Static variable to track the number of layers created, used for debugging purposes.

    """
    A Layer is a matrix with three properties: its dimensions n and m, and an activation function.
    Terminology: the p ('product') matrix is the product between the previous layer and weights, the a
    ('activation') matrix is the p matrix that has an activation function applied to it.
    
    A Layer should know the shape of the inputted matrix, the expected shape of the outputted matrix,
    and the activation function the programmer intends to use.
    """
    
    #TODO: handle checking for valid shapes
    """inputM and inputN are the dimensions of the inputted matrix, neurons specify the number of columns in the output matrix."""
    def __init__(self, inputM: int, inputN: int, neurons: int, funcName: str): #TODO: what if funcName is invalid?
        self.activationFunc  = ACTIVATION_FUNCS[funcName]
        self.activationDeriv = DERIV_ACTIVATION_FUNCS[f"{funcName}_d"]
        
        self.weights = 10 * np.random.rand(inputN, neurons)
        self.biases  = np.random.rand(inputM, neurons)
        
        self.p = np.zeros((inputM, neurons))
        self.a = np.zeros((inputM, neurons))

        self.layerIndex = Layer.index
        Layer.index += 1

    """Forward propagation algorithm: returns a numpy array of matrix multiplication and an applied activation
    function."""
    #Optimization: Fix: implement mini-batching or process per-sample; ensure shapes reflect (batch_size, features).
    def forward(self, input: ndarray) -> ndarray:
        #matrix multiplication
        self.p = input @ self.weights + self.biases
        self.a = self.activationFunc(self.p)

        #Printing for debugging... or visualization?
        if self.layerIndex == 0:
            print("Beginning forward propagation...")
        print(f"-----------------------Layer {self.layerIndex}-----------------------")
        print(f"input: {input}\n")
        print(f"weights: {self.weights}\n")
        print(f"biases: {self.biases}\n")
        print(f"p: {self.p}\n")
        print(f"a: {self.a}\n")

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

n = 8000 #Specify the number of rows to extract
trainInputs = dataDF[["x", "y"]].iloc[:n].to_numpy()
trainOutputs = dataDF[["z"]].iloc[:n].to_numpy()
a = dataDF[["x", "y"]]

test = dataDF[["x", "y"]].iloc[n:].to_numpy()
a = BasicANN(trainInputs)
epochs = 3
# out = None
for i in range(epochs):
    out = a.forwardPropagation()
    a.backPropagation(trainOutputs, 0.1)

    print(f"********************Epoch {i + 1} Results********************")
    print(f"Inputs: {trainInputs}")
    print(f"Predicted: {out}")
    print(f"Actual: {trainOutputs}\n")
