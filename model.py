#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import numpy as np
from numpy import ndarray

class Logger: pass

from activations import *
from losses import *
#TODO: add seed parameter instead of hardcoding seed
class ANN:
    """
    ANN initializes a multi layer perceptron. This model expects x,y pairs represented as numpy
    matrices of shape (n, 2) and produces a z-value for each pair (matrix of shape (n, 1)). The
    integer n represents the number of samples.
    
    Keyword arguments:
    
        - input_features (int): The number of columns in the training dataset used for
        inputs (no actual values for loss). Default = 2
        - batch_size (int): The number of samples to train on per each epoch. Default = 32
        - output_features (list[int]): Represent the number of features/neurons per layer.
        Default = [1, 1]
        - activations (list[ActivationFunction]): Represent the activation used per layer.
        Default = [ReLU, Identity]

    Default architecture:

        - 1st layer: 1 neuron with ReLU activation
        - 2nd/output layer: 1 neuron with Identity activation
    """

    def __init__(self, **kwargs) -> None:
        numFeatures = 2
        if (kwargs["input_features"] != None):
            numFeatures = kwargs["input_features"]

        batchSize = 32
        if (kwargs["batch_size"] != None):
            batchSize = kwargs["batch_size"]

        #TODO: how should this behave if there is a discrepancy between list sizes, or if only one list is inputted?
        numNeurons = [1, 1]
        if (kwargs["output_features"] != None):
            numNeurons = kwargs["output_features"]

        activations = [ReLU(), Identity()]
        if (kwargs["activations"] != None):
            activations=kwargs["activations"]
        
        self.batchSize = batchSize

        self.layers = []
        self.__addLayers(numFeatures, numNeurons, activations)

    def __addLayers(self, features: int, neuronsPerLayer: list[int], activationsPerLayer: list[ActivationFunction]) -> None:
        if (len(neuronsPerLayer) == len(activationsPerLayer)):
            n = 0
            for l in range(len(neuronsPerLayer)):
                if (len(self.layers) == 0 and l == 0): #If there are no layers, build starting w/ input
                    n = features
                else:
                    _, n = self.layers[l - 1].getAOutputs().shape

                self.layers.append(Layer(numInputFeatures=n, numOutputFeatures=neuronsPerLayer[l], layerSize=1000, activation=activationsPerLayer[l]))

    #TODO
    # def clear(): pass

    def forwardPropagation(self, input: ndarray) -> ndarray:
        i = 0
        a = input

        for i in range(len(self.layers)):
            layer = self.layers[i]
            a = layer.forward(a)

        return a #Return the output of the final layer
    
    #TODO:
    #   -enforce proper dimensionality of expected argument
    def backPropagation(self, input: ndarray, expected: ndarray, learnRate: float, lossFunc: LossFunction):
        self.__backPropagation(input, expected, learnRate, lossFunc)

    def __backPropagation(self, input: ndarray, expected: ndarray, learnRate: float, lossFunc: LossFunction):
        """Uses a rolling variable to track gradients. Every current layer updates the gradient,
        and the layers higher up the chain will reuse this gradient, updating it for each
        pass backwards towards the input layer.
        """
        start = len(self.layers)
        predicted = self.layers[start - 1].getAOutputs()
        #FIXME: Some loss functions may return a float instead of ndarray
        dEdPredicted = lossFunc.getDeriv()(predicted, expected)
        gradient = None
        updates = []

        for i in reversed(range(start)):
            #Initializing the current layer, its stored un-activated p values, and activation derivative
            layer = self.layers[i]
            p = layer.getPOutputs()
            actDeriv = layer.getActivation().getDeriv()

            if (i == start - 1): #Initializing gradient; this happens once at the start of the loop.
                gradient = np.multiply(dEdPredicted, actDeriv(p))
            else: #Updating gradient; this happens every time after the above condition.
                dPdAPrev = self.layers[i + 1].getWeights().T
                dAPrevdPPrev = actDeriv(p)
                gradient = np.multiply(gradient @ dPdAPrev, dAPrevdPPrev) #(gradient @ dPdAPrev) * dAPrevdPPrev
                
            #Initializing current layer's a value. Note that the rolling gradient does NOT get updated with a
            a = input.T
            if (i != 0):
                a = self.layers[i - 1].getAOutputs().T

            #Storing weight and bias updates. If the weights and biases were updated in this loop, it would cause improper gradient updates for preceding layers.
            dEdW = a @ gradient
            dEdB = np.mean(gradient, axis=0, keepdims=True)
            updates.append((dEdW, dEdB))
        
        for i in reversed(range(start)):
            layer = self.layers[i]
            dEdW, dEdB = updates[start - 1 - i]
            layer.updateParameters(dEdW, dEdB, learnRate)

    def __display(self, epoch: int, batchNum: int, predicted: ndarray, actual: ndarray):
        """Prints training results: epoch, predicted, actual, residuals, and error."""
        print(f"********************Epoch {epoch}, Batch {batchNum} Results********************")
        print(f"Predicted: {predicted}")
        print(f"Actual: {actual}")
        print(f"Residuals: {predicted - actual}")
        # print(f"Error: {self.getError()}\n")

    # def __saveTo():
    #     pass

    #FIXME: add logic to handle vectors
    def train(self, input: ndarray, expected: ndarray, learnRate: float, epochs: int, lossFunc: LossFunction, displayOutputs = False) -> list:
        """
        Uses gradient descent as the optimizer. If self.batchSize == the number of samples,
        this becomes stochastic gradient descent; otherwise, the optimizer is batch gradient
        descent by default.
        """

        shuffledInput, shuffledExpected = self.__shuffle(input, expected)
        samples, _ = shuffledInput.shape
        losses = []
        
        for i in range(epochs):
            for j in range(0, samples, self.batchSize):
                inputBatch  = shuffledInput[j:j+self.batchSize] #TODO: Does numpy fill 0s for <32 features?
                targetBatch = shuffledExpected[j:j+self.batchSize]

                zhat = self.forwardPropagation(inputBatch)
                self.backPropagation(input=inputBatch, expected=targetBatch, learnRate=learnRate, lossFunc=lossFunc)
                losses.append(lossFunc.getFunc()(zhat, targetBatch))

                if displayOutputs:
                    self.__display(i + 1, round(j / self.batchSize) + 1, zhat, targetBatch)

            shuffledInput, shuffledExpected = self.__shuffle(shuffledInput, shuffledExpected)
        
        return losses
    
    def __shuffle(self, input: ndarray, expected: ndarray) -> tuple[ndarray, ndarray]:
        rng = np.random.default_rng(seed=1)
        data = np.concatenate((input, expected), axis=1)

        samples, features = input.shape
        _, outputFeatures = expected.shape

        shuffled         = rng.permuted(data, axis=0)
        shuffledInput    = shuffled[:, [0, features]].reshape(samples, features)
        shuffledExpected = shuffled[:, features].reshape(samples, outputFeatures)

        return shuffledInput, shuffledExpected

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
                    # np.savetxt(f, self.getError(), "%d", ",", header="Error")

                f.close()
        
        return a

import math
import activations
class Layer:
    """
    A Layer transforms the data inputted to it with weights, biases, and an activation function.

    Parameters:

        - numInputFeatures: Represents the number of columns of the inputted matrix.
        - numOutputFeatures: Specifies the number of columns of the outputted matrix.
        - layerSize: Specifies the capacity of each layer.
        - activation: The ActivationFunction object that will be applied to the input.

    Instance variables:

        - self.p: The p ('product') matrix is the matrix multiplication product between the previous
        layer and weights. (Normal convention follows z, but due to the task of outputting a predicted
        Z-value, I chose p for the internal representation.)
        - self.a: The a ('activation') matrix is the p matrix that has an activation function
        applied to it.
        - self.activation: Holds the ActivationFunction object that has a method returning the callable
        version OF the activation. I chose to pass an object instead of a callable because it allows
        for type-checking for optimal weight initialization.

    """
    index = 0 #Static variable to track the number of layers created, used for debugging purposes.
    rng = np.random.default_rng(seed=1) #Static random number generator

    #TODO: handle checking for valid shapes
    def __init__(self, numInputFeatures: int, numOutputFeatures: int, layerSize: int, activation: ActivationFunction):
        self.activation = activation
        self.weights = Layer.rng.random((numInputFeatures, numOutputFeatures))
        self.biases  = np.zeros((1, numOutputFeatures))
        
        # if (type(self.activation) == activations.Sigmoid): #Normal Glorot initialization
        #     sd = math.sqrt(2 / (samples + samples)) #FIXME: this needs to take in a param for output samples instead of samples + samples
        #     self.weights *= sd
        if (type(self.activation) == activations.ReLU): #Normal He initialization
            sd = math.sqrt(2 / numOutputFeatures)
            self.weights *= sd
        #TODO: uniform He init

        #FIXME: this changes nothing
        self.p = np.zeros((layerSize, numOutputFeatures))
        self.a = np.zeros((layerSize, numOutputFeatures))

        self.layerIndex = Layer.index
        Layer.index += 1

    #TODO: how do I handle input's shape mismatching with initialized p and a matrices?
    #TODO: display outputs in a logger class
    def forward(self, input: ndarray, displayParams = False) -> ndarray:
        """
        Forward propagation algorithm: returns a numpy array of matrix multiplication 
        between input and weights plus a bias, applied to an activation function.
        """
        
        self.p = input @ self.weights + self.biases #matrix multiplication and addition
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
    
    def updateParameters(self, dw: ndarray, db: ndarray, learnRate: int):
        """
        Backward propagation to update weights and biases. No fancy optimization
        techniques... yet.
        """
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
