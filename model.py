#Author: Pranangel
#Purpose: Has the building blocks for a customizable artificial neural network.

from numpy import ndarray
import numpy as np
import math
from initializers import INITIALIZERS
from activations import ACTIVATIONS
from losses import LOSSES

#NOTE: these are only for reproducibility in testing
LAYER_RNG    = np.random.default_rng(seed=1)
TRAINING_RNG = np.random.default_rng(seed=2)

class Layer:
    """
    A Layer transforms the data inputted to it with weights, biases, and an activation function.

    Parameters:

        - numInputFeatures: Represents the number of columns of the inputted matrix. (Same as fan in.)
        - numOutputFeatures: Specifies the number of columns of the outputted matrix. (Same as fan out.)
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

    #TODO: handle checking for valid shapes
    def __init__(self, numInputFeatures: int, numOutputFeatures: int, activationName: str, initializerName=""):
        self.numInputFeatures  = numInputFeatures
        self.numOutputFeatures = numOutputFeatures
        self.activationName    = activationName
        self.initializerName   = initializerName

        self.activation = ACTIVATIONS[self.activationName]
        self.weights = np.ones((self.numInputFeatures, self.numOutputFeatures))
        self.biases  = np.zeros((1, self.numOutputFeatures))
        self.velocity =  {
            "weight_gradient": np.zeros(self.weights.shape),
            "bias_gradient": np.zeros(self.biases.shape)
        }

        if (self.activationName == "sigmoid"):
            self.weights = INITIALIZERS["xavier_uniform"].initialize(numInputFeatures, numOutputFeatures)

        if ("relu" in self.activationName):
            self.weights = INITIALIZERS["he_uniform"].initialize(numInputFeatures, numOutputFeatures, seed=2)
        
        if not (initializerName == "" or initializerName == '' or (initializerName is None)):
            self.weights = INITIALIZERS[initializerName].initialize(numInputFeatures, numOutputFeatures)

        self.p: ndarray
        self.a: ndarray

        self.layerIndex = Layer.index
        Layer.index += 1

    def forward(self, input: ndarray) -> ndarray:
        """
        Forward propagation algorithm: returns a numpy array of matrix multiplication 
        between input and weights plus a bias, applied to an activation function.
        """
        self.p = input @ self.weights + self.biases #matrix multiplication and addition
        self.a = self.activation.getFunc(self.p)

        return self.a

    def updateParameters(self, dw: ndarray, db: ndarray, learnRate: int):
        """
        Backward propagation to update weights and biases. Uses simple momentum optimization.
        """

        momentum = 0.01
        weightVelocity = (momentum * self.velocity["weight_gradient"]) + dw
        biasVelocity   = (momentum * self.velocity["bias_gradient"]) + db

        self.weights = self.weights - (weightVelocity * learnRate)
        self.biases  = self.biases - (biasVelocity * learnRate)

        self.velocity["weight_gradient"] = weightVelocity
        self.velocity["bias_gradient"]   = biasVelocity

    def resetParameters(self):
        self.weights = np.ones(self.weights.shape)
        self.biases  = np.zeros(self.biases.shape)
        self.velocity =  {
            "weight_gradient": np.zeros(self.weights.shape),
            "bias_gradient": np.zeros(self.biases.shape)
        }

        fanIn, fanOut = self.weights.shape

        if (self.activationName == "sigmoid"):
            self.weights = INITIALIZERS["xavier_uniform"].initialize(fanIn, fanOut)

        if ("relu" in self.activationName):
            self.weights = INITIALIZERS["he_uniform"].initialize(fanIn, fanOut, seed=2)
        
        if not (self.initializerName == "" or self.initializerName == '' or (self.initializerName is None)):
            self.weights = INITIALIZERS[self.initializerName].initialize(fanIn, fanOut)

        self.p = np.zeros(self.p.shape)
        self.a = np.zeros(self.a.shape)


class ANN:
    """
    ANN initializes a multi layer perceptron. This model expects x,y pairs represented as numpy
    matrices of shape (n, 2) and produces a z-value for each pair (matrix of shape (n, 1)). The
    integer n represents the number of samples.

    
    A user must initialize any ANN with the number of input features. If they choose to pre-
    initialize, they have the option of specifying the output_features and activations kwargs
    (see below), or no kwargs. If no kwargs are passed, a default model is constructued (see
    below.)
    
    Keyword Arguments::
    
        - output_features (list[int]): Represent the number of features/neurons per layer.
        Default = [1, 1]
        - activations (list[str]): Represent the activation used per layer.
        Default = ["relu", "identity"]

    Default Architecture::

        - 1st layer: 100 neurons (fan out) with ReLU activation and uniform He weight init
        - 2nd layer: 100 neurons (fan out) with ReLU activation and uniform He weight init
        - 3rd layer: 1 neuron (fan out) with Identity activation

    Note:: ANN only supports simple momentum optimization, which is used by default.
    """

    def __init__(self, inputFeatures: int, preinitialize=False, **kwargs):
        self.numInputFeatures = inputFeatures
        self.layers = []

        if (preinitialize):
            neuronsPerLayer = []
            neuronsPerLayer: list[int]
            activationsPerLayer = []
            activationsPerLayer: list[str]

            if (kwargs["output_features"] != None and
                kwargs["activations"]     != None
            ):
                neuronsPerLayer     = kwargs["output_features"]
                activationsPerLayer = kwargs["activations"]

                self.__addLayers(neuronsPerLayer, activationsPerLayer)

            elif (kwargs == None):
                neuronsPerLayer = [100, 100, 1]
                activationsPerLayer = ["relu", "relu", "identity"]

                self.__addLayers(neuronsPerLayer, activationsPerLayer)

            else:
                print("Error: bad init")

            
    def add(self, **kwargs):
        if (kwargs["output_features"] != None and kwargs["activation"] != None):
            n = self.numInputFeatures
            if (len(self.layers) > 0):
                _, n = self.layers[len(self.layers) - 1].weights.shape
            self.layers.append(Layer(numInputFeatures=n, numOutputFeatures=kwargs["output_features"], activationName=kwargs["activation"]))
        else:
            if (kwargs["output_features"] == None and kwargs["activation"] == None):
                print("Error: arg 'output_features' and arg 'activation' not supplied.")
            elif (kwargs["output_features"] == None):
                print("Error: arg 'output_features' not supplied.")
            elif (kwargs["activation"] == None):
                print("Error: arg 'activation' not supplied.")
            else:
                print("Error.")

    def forwardPropagation(self, input: ndarray) -> ndarray:
        i = 0
        a = input

        for i in range(len(self.layers)):
            layer = self.layers[i]
            a = layer.forward(a)

        return a #Return the output of the final layer
    
    def backPropagation(self, input: ndarray, expected: ndarray, learnRate: float, lossFuncName: str):
        self.__backPropagation(input, expected, learnRate, lossFuncName)

    #FIXME: add logic to handle vectors
    def train(self, input: ndarray, expected: ndarray, learnRate: float, epochs: int, batchSize: int, lossFuncName: str, displayOutputs = False) -> list:
        """
        Uses mini-batch gradient descent as the default optimizer. If batchSize == 1, this becomes stochastic
        gradient descent; if batchSize == samples, this becomes batch gradient descent.
        """

        shuffledInput, shuffledExpected = self.__shuffle(input, expected)
        samples, _ = shuffledInput.shape
        lossFunc = LOSSES[lossFuncName]
        avgLosses = []
        
        for i in range(epochs):
            losses = []

            for j in range(0, samples, batchSize):
                inputBatch  = shuffledInput[j:j + batchSize]
                targetBatch = shuffledExpected[j:j + batchSize]

                zHat = self.forwardPropagation(inputBatch)
                self.backPropagation(input=inputBatch, expected=targetBatch, learnRate=learnRate, lossFuncName=lossFuncName)
                losses.append(lossFunc.getLoss(zHat, targetBatch))

                if displayOutputs:
                    self.__display(i + 1, (j // batchSize) + 1, zHat, targetBatch)

            avgLosses.append(np.mean(losses))
            shuffledInput, shuffledExpected = self.__shuffle(shuffledInput, shuffledExpected)
        
        return avgLosses

    def test(self, testInput: ndarray) -> ndarray:
        return self.forwardPropagation(testInput)

    def resetParameters(self):
        for layer in self.layers:
            layer.resetParameters()
    
    def __addLayers(self, neuronsPerLayer: list[int], activationsPerLayer: list[str]):
        if (len(neuronsPerLayer) == len(activationsPerLayer)):
            n = 0
            for l in range(len(neuronsPerLayer)):
                if (len(self.layers) == 0 and l == 0): #If there are no layers, build starting w/ input
                    n = self.numInputFeatures
                else:
                    _, n = self.layers[l - 1].weights.shape

                self.layers.append(Layer(numInputFeatures=n, numOutputFeatures=neuronsPerLayer[l], activationName=activationsPerLayer[l]))
        else:
            print(f"Error: dimension mismatch in arg 1 ({len(neuronsPerLayer)}) and arg 2 ({len(activationsPerLayer)})")

    def __backPropagation(self, input: ndarray, expected: ndarray, learnRate: float, lossFuncName: str):
        """
        Uses a rolling variable to track gradients. Every current layer updates the gradient,
        and the layers higher up the chain will reuse this gradient, updating it for each
        pass backwards towards the input layer.
        """
        lossFunc = LOSSES[lossFuncName]
        start = len(self.layers)
        predicted = self.layers[start - 1].a
        #FIXME: Some loss functions may return a float instead of ndarray
        dEdPredicted = lossFunc.getDeriv(predicted, expected)
        gradient = None
        updates = []

        for i in reversed(range(start)):
            #Initializing the current layer, its stored un-activated p values, and activation derivative
            layer = self.layers[i]
            p = layer.p
            actDeriv = layer.activation.getDeriv(p)

            if (i == start - 1): #Initializing gradient; this happens once at the start of the loop.
                gradient = np.multiply(dEdPredicted, actDeriv)
            else: #Updating gradient; this happens every time after the above condition.
                dPdAPrev = self.layers[i + 1].weights.T
                dAPrevdPPrev = actDeriv
                gradient = np.multiply(gradient @ dPdAPrev, dAPrevdPPrev) #(gradient @ dPdAPrev) * dAPrevdPPrev
                
            #Initializing current layer's a value. Note that the rolling gradient does NOT get updated with a
            a = input.T
            if (i != 0):
                a = self.layers[i - 1].a.T

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
        
    def __shuffle(self, input: ndarray, expected: ndarray) -> tuple[ndarray, ndarray]:
        data = np.concatenate((input, expected), axis=1)

        samples, features = input.shape
        _, outputFeatures = expected.shape

        shuffled         = TRAINING_RNG.permutation(data)
        shuffledInput    = shuffled[:, :features].reshape(samples, features)
        shuffledExpected = shuffled[:, features:].reshape(samples, outputFeatures)

        return shuffledInput, shuffledExpected
