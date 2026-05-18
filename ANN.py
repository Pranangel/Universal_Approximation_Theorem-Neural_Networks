#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import numpy as np
from numpy import ndarray

class Loss: pass

class Model: pass

class Trainer: pass

class Tester: pass

from activations import *
from errors import *
class BasicANN:
    """
    BasicANN initializes a predefined artificial neural network. This model expects (x,y) pairs
    (matrix of shape n, 2) and produces a z-value for each pair (matrix of shape n, 1).
    
    Default architecture:
    1st layer: 1 neuron with ReLU activation
    2nd/output layer: 1 neuron with sigmoid activation
    Loss function: Mean Squared Error
    """
    
    #FIXME: If the programmer wants to build their own, they must pass ALL OF numLayers, numNeurons, AND activations.
    def __init__(self, numFeatures: int, batchSize: int, numLayers: int, numNeurons: list[int], activations:list[ActivationFunction]) -> None:
        if (batchSize   == None): batchSize = 32
        if (numLayers   == None): numLayers = 2
        #TODO: how should this behave if there is a discrepancy between list sizes, or if only one list is inputted?
        if (numNeurons  == None): numNeurons = [1,1]
        if (activations == None): activations=[ActivationFunction.RELU, ActivationFunction.SIGMOID]
        
        self.features  = numFeatures
        self.batchSize = batchSize
        self.input: ndarray
        self.error: ndarray

        self.layers       = []
        self.addLayers(numLayers, numNeurons, activations)

    def addLayers(self, numLayers: int, neuronsPerLayer: list[int], activationsPerLayer: list[ActivationFunction]) -> None:
        if (numLayers == len(neuronsPerLayer)) and (numLayers == len(activationsPerLayer)):
            m, n = 0, 0
            for l in range(numLayers):
                if (len(self.layers) == 0 and l == 0): #If there are no layers, build starting w/ input
                    m = self.batchSize
                    n = self.features
                else:
                    m, n = self.layers[l - 1].getAOutputs().shape

                self.layers.append(Layer(inputM=m, inputN=n, neurons=neuronsPerLayer[l], activation=activationsPerLayer[l]))

    #TODO
    # def clear(): pass

    def forwardPropagation(self, input: ndarray) -> ndarray:
        i = 0
        a = input
        for i in range(len(self.layers)):
            layer = self.layers[i]
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
    def backPropagation(self, input: ndarray, z: ndarray, learnRate: float, errorFunc: ErrorFunction):
        self.__backPropagation(input, z, learnRate, errorFunc)

    def __backPropagation(self, input: ndarray, z: ndarray, learnRate: float, errorFunc: ErrorFunction):
        """Uses a rolling variable to track gradients. Every current layer updates the gradient,
        and the layers higher up the chain will reuse this gradient, updating it for each
        pass backwards towards the input layer.
        """
        last = len(self.layers) - 1
        zPredicted = self.layers[last].getAOutputs()
        #FIXME: Some functions may return a float instead of ndarray
        self.error = errorFunc.getFunc()(zPredicted, z) #FIXME: should self.error be an instance member at all?
        dEdZPredicted = errorFunc.getDeriv()(zPredicted, z)
        gradient = np.zeros(input.shape)

        for i in reversed(range(last)):
            layer = self.layers[i]

            p = layer.getPOutputs()
            actFunc = layer.getActivation().getDeriv()
            a = input.T
            if (i != 0):
                a = self.layers[i - 1].getAOutputs().T #rolling gradient does NOT get updated with a

            if (i == 0):
                gradient = np.multiply(dEdZPredicted, actFunc(p))
            else:
                dPdAPrev = self.layers[i + 1].getWeights().T
                dAPrevdPPrev = actFunc(p)
                gradient = np.multiply(gradient @ dPdAPrev, dAPrevdPPrev) #(gradient @ dPdAPrev) * dAPrevdPPrev

            dEdW = a @ gradient
            dEdB = np.sum(gradient, axis=0, keepdims=True) #TODO: update by mean instead of sum
            self.layers[i].updateParameters(dEdW, dEdB, learnRate)

    def getError(self) -> ndarray:
        return self.error

    def __display(self, epoch: int, batchNum: int, predicted: ndarray, actual: ndarray):
        """Prints training results: epoch, predicted, actual, residuals, and error."""
        print(f"********************Epoch {epoch}, Batch {batchNum} Results********************")
        print(f"Predicted: {predicted}")
        print(f"Actual: {actual}")
        print(f"Residuals: {predicted - actual}")
        print(f"Error: {self.getError()}\n")

    # def __saveTo():
    #     pass

    """
    Batch gradient descent.
    """
    # If this was stochastic, it would take the whole training dataset and inside of the epoch loop,
    # there would be another loop that does forward and backward for each point in the dataset
    #FIXME: safe file writing
    def train(self, input: ndarray, z: ndarray, learnRate: float, epochs: int, errorFunc: ErrorFunction, displayOutputs = False, saveFile = ""):
        samples, _ = input.shape
        
        for i in range(epochs):
            for j in range(0, samples, self.batchSize):
                inputBatch = input[j:j+self.batchSize] #TODO: Does numpy fill 0s for <32 features?
                predBatch  = z[j:j+self.batchSize]
                a = self.forwardPropagation(inputBatch)
                self.backPropagation(input=inputBatch, z=predBatch, learnRate=learnRate, errorFunc=errorFunc)
                self.__display(i + 1, round(j / self.batchSize) + 1, a, predBatch)

    def test(self, testInput: ndarray, displayPredictions=False, saveFile = "", testOutput = None) -> ndarray:
        a = self.forwardPropagation(testInput)

        if (displayPredictions):
            print("-------------------------------TESTING-------------------------------")
            print(f"Inputs: {testInput}")
            print(f"Predicted: {a}")

            if (type(testOutput) == ndarray):
                print(f"Actual: {testOutput}")

        if (saveFile != "" and saveFile != None):
            with open(saveFile, "a") as f:
                np.savetxt(f, testInput, "%d", ",", header="Inputs")
                np.savetxt(f, a, "%d", ",", header="Predicted")

                if (type(testOutput) == ndarray):
                    np.savetxt(f, testOutput, "%d", ",", header="Actual")
                    np.savetxt(f, a - testOutput, "%d", ",", header="Residuals")
                    np.savetxt(f, self.getError(), "%d", ",", header="Error")

                f.close()
        
        return a

"""
A Layer is a matrix with three properties: its dimensions n and m, and an activation function.
Terminology: the p ('product') matrix is the product between the previous layer and weights, the a
('activation') matrix is the p matrix that has an activation function applied to it.

A Layer expects the shape of the inputted matrix, the neurons/columns that will be stored,
and the activation function's name.
"""
class Layer:
    index = 0 #Static variable to track the number of layers created, used for debugging purposes.

    #TODO: handle checking for valid shapes
    """inputM and inputN are the dimensions of the inputted matrix, neurons specify the number of columns in the output matrix."""
    def __init__(self, inputM: int, inputN: int, neurons: int, activation: ActivationFunction): #TODO: what if funcName is invalid?
        self.activation      = activation
        
        #TODO: Xavier/Glorot initialization for sigmoid, He initialization for ReLU
        self.weights = np.random.rand(inputN, neurons)
        self.biases  = np.random.rand(1, neurons)
        
        #FIXME: Layer size should be independent of batch size
        self.p = np.zeros((inputM, neurons))
        self.a = np.zeros((inputM, neurons))

        self.layerIndex = Layer.index
        Layer.index += 1

    """Forward propagation algorithm: returns a numpy array of matrix multiplication and an applied activation
    function."""
    #TODO: ensure shapes reflect (batch_size, features).
    def forward(self, input: ndarray, displayParams = False) -> ndarray:
        #matrix multiplication
        self.p = input @ self.weights + self.biases
        self.a = self.activation.getFunc()(self.p)

        if (displayParams):
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

    def getActivation(self) -> ActivationFunction:
        return self.activation

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

n = 10 #Specify the number of rows to extract for training and testing
trainInputs = dataDF[["x", "y"]].iloc[:n].to_numpy()
trainOutputs = dataDF[["z"]].iloc[:n].to_numpy()

#TODO: generate non-normalized data and compare results to normalized data
model1 = BasicANN(numFeatures=2, batchSize=2, numLayers=2, numNeurons=[10, 1], activations=[Sigmoid(), Sigmoid()])
model1.train(input=trainInputs, z=trainOutputs, learnRate=0.1, epochs=10, errorFunc=MeanSquaredError(), displayOutputs=True)

#TODO: add visualization to training and testing
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

#Testing
test = dataDF[["x", "y"]].iloc[n:5001].to_numpy()
predictions = model1.test(testInput=test, displayPredictions=True)
print(f"Actual: {dataDF[["z"]].iloc[n:5001].to_numpy()}") #cheating
