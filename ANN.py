#Author: Pranangel
#Purpose: Making the building blocks for a customizable artificial neural network.

#https://medium.com/analytics-vidhya/activation-functions-optimization-techniques-and-loss-functions-75a0eea0bc31
import numpy as np
from numpy import ndarray

class Activations: #TODO
    def __init__(self): pass

class Losses: #TODO
    def __init__(self) -> None:
        pass

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
    positiveMask = x > 0
    result = np.zeros_like(a=x, shape=x.shape)
    result[positiveMask] = 1

    return result

#FIXME
# def softmax(mat: ndarray) -> ndarray:
#     """Takes a matrix as an argument and calculates a probability distribution 
#     represented by a matrix with the same shape."""

#     sumExp = np.sum(np.exp(mat))
#     return np.exp(mat) / sumExp

ACTIVATION_FUNCS = {
    "sigmoid" : sigmoid,
    "relu"    : relu,
    #"softmax" : softmax
}

DERIV_ACTIVATION_FUNCS = {
    "sigmoid_d" : derivSigmoid,
    "relu_d"    : derivRelu,
}

#TODO: log cosh loss

def squaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return (predicted - actual) ** 2

def derivSquaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return 2 * (predicted - actual)

#TODO: add per sample (axis=1), per output (axis=0)
def meanSquaredError(predicted: ndarray, actual: ndarray): #outputs floating[Any]
    return np.mean((predicted - actual) ** 2)

def derivMeanSquaredError(predicted: ndarray, actual: ndarray) -> ndarray:
    return 2 * (predicted - actual) / predicted.size

#FIXME
# def meanAbsError(predicted: ndarray, actual: ndarray) -> ndarray:
#     m1, n1 = predicted.shape #TODO: check for same-size shape
#     m2, n2 = actual.shape

#     return np.abs(predicted - actual)  / n1

# def derivMeanAbsError(predicted: ndarray, actual: ndarray) -> ndarray:
#     m1, n1 = predicted.shape #TODO: check for same-size shape
#     m2, n2 = actual.shape

#     diff = predicted - actual
#     copy = np.ones((m1, m2))

#     if (diff > 0):
#         return copy * (1 / n1)
#     elif (diff < 0):
#         return copy * (-1 / n1)
#     return copy * 0

#TODO: Add binary cross entropy
ERROR_FUNCS = {
    "SE"  : squaredError,
    "MSE" : meanSquaredError,
    # "MAE" : meanAbsError
}

DERIV_ERROR_FUNCS = {
    "SE_d"  : derivSquaredError,
    "MSE_d" : derivMeanSquaredError,
    # "MAE_d" : derivMeanAbsError
}

class Activation: 

    def __init__(self):
        pass

class Loss: pass

class Model: pass

class Trainer: pass

class Tester: pass

class BasicANN:
    """
    BasicANN initializes a predefined artificial neural network. This model expects (x,y) pairs
    (matrix of shape n, 2) and produces a z-value for each pair (matrix of shape n, 1).
    
    Default architecture:
    1st layer: 1 neuron with sigmoid activation
    2nd/output layer: 1 neuron with sigmoid activation
    Loss function: Mean Squared Error
    """
    
    #TODO: If the programmer wants to build their own, they must pass ALL OF numLayers, numNeurons, AND activations.
    def __init__(self, numFeatures: int, batchSize=None, numLayers=2, numNeurons=[1, 1], activations=["sigmoid", "sigmoid"]) -> None:
        if (batchSize == None): batchSize = 32 #FIXME
        
        self.features  = numFeatures
        self.batchSize = batchSize
        self.input: ndarray
        self.error: ndarray

        #TODO: store in dictionary?
        self.layers       = []
        self.weights      = []
        self.biases       = []
        self.pOutputs     = []
        self.aOutputs     = []
        self.activations  = []

        self.addLayers(numLayers, numNeurons, activations)

    def addLayers(self, numLayers: int, neuronsPerLayer: list[int], activationsPerLayer: list[str]) -> None:
        if (numLayers == len(neuronsPerLayer)) and (numLayers == len(activationsPerLayer)):
            m, n = 0, 0
            for l in range(numLayers):
                if (len(self.layers) == 0 and l == 0): #If there are no layers, build starting w/ input
                    m = self.batchSize
                    n = self.features
                else:
                    m, n = self.layers[l - 1].getAOutputs().shape

                self.layers.append(Layer(inputM=m, inputN=n, neurons=neuronsPerLayer[l], funcName=activationsPerLayer[l]))

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
    #   -Check valid lossFunc string
    def backPropagation(self, input: ndarray, z: ndarray, learnRate: float, lossFuncName: str):
        self.__backPropagation(input, z, learnRate, lossFuncName)

    #TODO
    def __backPropagation(self, input: ndarray, z: ndarray, learnRate: float, lossFuncName: str):
        """Uses a rolling variable to track gradients. Every current layer updates the gradient,
        and the layers higher up the chain will reuse this gradient, updating it for each
        pass backwards towards the input layer.
        """
        last = len(self.layers) - 1
        zPredicted = self.layers[last].getAOutputs()
        self.error = ERROR_FUNCS[lossFuncName](zPredicted, z) #FIXME: should self.error be an instance member at all?
        dEdZPredicted = DERIV_ERROR_FUNCS[f"{lossFuncName}_d"](zPredicted, z)
        gradient = np.zeros(input.shape)

        for i in reversed(range(last)):
            layer = self.layers[i]

            p = layer.getPOutputs()
            actFunc = layer.getActivationDeriv()
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

    If this was stochastic, it would take the whole training dataset and inside of the epoch loop,
    there would be another loop that does forward and backward for each point in the dataset
    """
    #FIXME: safe file writing
    def train(self, input: ndarray, z: ndarray, learnRate: float, epochs: int, lossFuncName: str, displayOutputs = False, saveFile = ""):
        samples, _ = input.shape
        
        for i in range(epochs):
            for j in range(0, samples, self.batchSize):
                inputBatch = input[j:j+self.batchSize] #TODO: Does numpy fill 0s for <32 features?
                predBatch  = z[j:j+self.batchSize]
                a = self.forwardPropagation(inputBatch)
                self.backPropagation(input=inputBatch, z=predBatch, learnRate=learnRate, lossFuncName=lossFuncName)
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

class Layer:
    index = 0 #Static variable to track the number of layers created, used for debugging purposes.

    """
    A Layer is a matrix with three properties: its dimensions n and m, and an activation function.
    Terminology: the p ('product') matrix is the product between the previous layer and weights, the a
    ('activation') matrix is the p matrix that has an activation function applied to it.
    
    A Layer expects the shape of the inputted matrix, the neurons/columns that will be stored,
    and the activation function's name.
    """
    
    #TODO: handle checking for valid shapes
    """inputM and inputN are the dimensions of the inputted matrix, neurons specify the number of columns in the output matrix."""
    def __init__(self, inputM: int, inputN: int, neurons: int, funcName: str): #TODO: what if funcName is invalid?
        self.activationFunc  = ACTIVATION_FUNCS[funcName]
        self.activationDeriv = DERIV_ACTIVATION_FUNCS[f"{funcName}_d"]
        
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
    #TODO: implement mini-batching or process per-sample; ensure shapes reflect (batch_size, features).
    def forward(self, input: ndarray, displayParams = False) -> ndarray:
        #matrix multiplication
        self.p = input @ self.weights + self.biases
        self.a = self.activationFunc(self.p)

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

n = 3200 #Specify the number of rows to extract for training and testing
trainInputs = dataDF[["x", "y"]].iloc[:n].to_numpy()
trainOutputs = dataDF[["z"]].iloc[:n].to_numpy()

model1 = BasicANN(numFeatures=2, batchSize=64, numLayers=2, numNeurons=[10, 1], activations=["sigmoid", "relu"])
#TODO: generate non-normalized data and compare results to normalized data
model1.train(input=trainInputs, z=trainOutputs, learnRate=0.1, epochs=100, lossFuncName="MSE", displayOutputs=True)

#TODO: add visualization to training and testing
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

#Testing
test = dataDF[["x", "y"]].iloc[n:5001].to_numpy()
predictions = model1.test(testInput=test, displayPredictions=True)
print(f"Actual: {dataDF[["z"]].iloc[n:5001].to_numpy()}") #cheating
