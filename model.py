#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

import numpy as np
from numpy import ndarray
from activations import *
from losses import *
# import random

#NOTE: these are only for reproducibility in testing
LAYER_RNG    = np.random.default_rng(seed=1)
TRAINING_RNG = np.random.default_rng(seed=2)

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
        - activations (list[str]): Represent the activation used per layer.
        Default = ["relu", "identity"]

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

        activations = ["relu", "identity"]
        if (kwargs["activations"] != None):
            activations=kwargs["activations"]
        
        self.batchSize = batchSize

        self.layers = []
        self.__addLayers(numFeatures, numNeurons, activations)

    #TODO: have users manually select num of output channels
    #TODO: make this public and have it take only a Layer parameter. The argument gets appended to the list
    def __addLayers(self, features: int, neuronsPerLayer: list[int], activationsPerLayer: list[str]) -> None:
        if (len(neuronsPerLayer) == len(activationsPerLayer)):
            n = 0
            for l in range(len(neuronsPerLayer)):
                if (len(self.layers) == 0 and l == 0): #If there are no layers, build starting w/ input
                    n = features
                else:
                    _, n = self.layers[l - 1].getAOutputs().shape

                self.layers.append(Layer(numInputFeatures=n, numOutputFeatures=neuronsPerLayer[l], layerSize=1000, activationName=activationsPerLayer[l]))

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
            actDeriv = layer.getActivationDeriv(p)

            if (i == start - 1): #Initializing gradient; this happens once at the start of the loop.
                gradient = np.multiply(dEdPredicted, actDeriv)
            else: #Updating gradient; this happens every time after the above condition.
                dPdAPrev = self.layers[i + 1].getWeights().T
                dAPrevdPPrev = actDeriv
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
                inputBatch  = shuffledInput[j:j + self.batchSize] #TODO: Does numpy fill 0s for <32 features?
                targetBatch = shuffledExpected[j:j + self.batchSize]

                zHat = self.forwardPropagation(inputBatch)
                self.backPropagation(input=inputBatch, expected=targetBatch, learnRate=learnRate, lossFunc=lossFunc)
                losses.append(lossFunc.getFunc()(zHat, targetBatch))

                if displayOutputs:
                    self.__display(i + 1, round(j / self.batchSize) + 1, zHat, targetBatch)
            #     break
            # break

            shuffledInput, shuffledExpected = self.__shuffle(shuffledInput, shuffledExpected)
        
        return losses
    
    def __shuffle(self, input: ndarray, expected: ndarray) -> tuple[ndarray, ndarray]:
        data = np.concatenate((input, expected), axis=1)

        samples, features = input.shape
        _, outputFeatures = expected.shape

        shuffled         = TRAINING_RNG.permutation(data)
        shuffledInput    = shuffled[:, :features].reshape(samples, features)
        shuffledExpected = shuffled[:, features:].reshape(samples, outputFeatures)

        return shuffledInput, shuffledExpected

    def test(self, testInput: ndarray) -> ndarray:
        return self.forwardPropagation(testInput) 

import math
import activations
class Layer:
    """
    A Layer transforms the data inputted to it with weights, biases, and an activation function.

    Parameters:

        - numInputFeatures: Represents the number of columns of the inputted matrix. (Same as fan in.)
        - numOutputFeatures: Specifies the number of columns of the outputted matrix. (Same as fan out.)
        - layerSize: Specifies the capacity of each layer.
        - activation: The ActivationFunction object that will be applied to the input.

    Instance variables:

        - self.p: The p ('product') matrix is the matrix multiplication product between the previous
        layer and weights. (Normal convention follows z, but due to the task of outputting a predicted
        Z-value, I chose p for the internal representation.)
        - self.a: The a ('activation') matrix is the p matrix that has an activation function
        applied to it.
        - self.activation: Holds the ActivationFunction object which has methods returning the
        activated input and the derivative activated input.

    """
    index = 0 #Static variable to track the number of layers created, used for debugging purposes.
    functionList = {
        "sigmoid": activations.Sigmoid(),
        "relu": activations.ReLU(),
        "leaky_relu": activations.LeakyReLU(),
        "identity": activations.Identity(),
        "": activations.Identity(),
        None: activations.Identity()
    }

    #TODO: handle checking for valid shapes
    def __init__(self, numInputFeatures: int, numOutputFeatures: int, layerSize: int, activationName: str, initializerName=""):
        self.activation = Layer.functionList[activationName]
        self.weights = np.ones((numInputFeatures, numOutputFeatures))#LAYER_RNG.random((numInputFeatures, numOutputFeatures))
        self.biases  = np.zeros((1, numOutputFeatures))
        self.velocity =  {
            "weight_gradient": np.zeros(self.weights.shape),
            "bias_gradient": np.zeros(self.biases.shape)
        }
        
        if (initializerName == "xavier_uniform" or activationName == "sigmoid"):
            sd = math.sqrt(6.0 / (numInputFeatures + numOutputFeatures))
            self.weights *= sd
        elif (initializerName == "xavier_normal"):
            sd = math.sqrt(2.0 / (numInputFeatures + numOutputFeatures))
            self.weights *= sd
        elif (initializerName == "he_uniform" or activationName == "relu"): #Generates negative values.
            limit = math.sqrt(6.0 / numInputFeatures) #NOTE: Numpy uses a half-open range, so the distribution is uniform from [min, max).
            self.weights = LAYER_RNG.uniform(low=-limit, high=limit, size=(numInputFeatures, numOutputFeatures))
        elif (initializerName == "he_normal"): #Values close to 0 have higher likelihood.
            sd = math.sqrt(2.0 / numOutputFeatures)
            self.weights *= sd

        #TODO: add a layersize cap for massive datasets
        self.p = np.zeros((layerSize, numOutputFeatures))
        self.a = np.zeros((layerSize, numOutputFeatures))

        self.layerIndex = Layer.index
        Layer.index += 1

    #TODO: how do I handle input's shape mismatching with initialized p and a matrices?
    #TODO: display outputs in a logger class
    def forward(self, input: ndarray) -> ndarray:
        """
        Forward propagation algorithm: returns a numpy array of matrix multiplication 
        between input and weights plus a bias, applied to an activation function.
        """
        self.p = input @ self.weights + self.biases #matrix multiplication and addition
        self.a = self.activation.getFunc(self.p)

        # print(f"Layer {self.layerIndex}:")
        # print(f"\tProportion activated: {np.mean(self.p > 0) * 100}%")
        # print(f"\tProportion negative weights: {np.mean(self.weights < 0) * 100}%")

        return self.a

    def updateParameters(self, dw: ndarray, db: ndarray, learnRate: int):
        """
        Backward propagation to update weights and biases. Uses simple momentum optimization.
        """

        if (not isinstance(self.activation, activations.Identity)):
            momentum = 0.3
            weightVelocity = (momentum * self.velocity["weight_gradient"]) + dw
            biasVelocity   = (momentum * self.velocity["bias_gradient"]) + db

            self.weights = self.weights - (weightVelocity * learnRate)
            self.biases  = self.biases - (biasVelocity * learnRate)

            self.velocity["weight_gradient"] = weightVelocity
            self.velocity["bias_gradient"]   = biasVelocity

    def getActivationDeriv(self, input: ndarray):
        return self.activation.getDeriv(input)

    def getWeights(self) -> ndarray:
        return self.weights
    
    def getBiases(self) -> ndarray:
        return self.biases
    
    def getPOutputs(self) -> ndarray:
        return self.p
    
    def getAOutputs(self) -> ndarray:
        return self.a

def puffle(input: ndarray, expected: ndarray) -> tuple[ndarray, ndarray]:
    data = np.concatenate((input, expected), axis=1)

    samples, features = input.shape
    _, outputFeatures = expected.shape

    shuffled         = TRAINING_RNG.permutation(data)
    shuffledInput    = shuffled[:, :features].reshape(samples, features)
    shuffledExpected = shuffled[:, features:].reshape(samples, outputFeatures)

    return shuffledInput, shuffledExpected
