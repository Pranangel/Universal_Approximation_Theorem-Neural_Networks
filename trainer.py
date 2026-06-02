#Author: Pranangel
#Purpose: File to train model and save predictions. Logger not included.

import pandas as pd
import numpy as np
from numpy import ndarray

from model import ANN
from activations import *
from losses import *


def linearScaling(data: pd.DataFrame) -> ndarray:
    """
    Normalizes in range [0, 1); formula taken from
    https://developers.google.com/machine-learning/crash-course/numerical-data/normalization
    
    This method expects the data parameter to have only 3 columns labelled "x", "y", and "z".

    training_data.csv ranges:
    x range: [-2.99981569, 2.99734622]
    y range: [-2.99905353, 2.99883086]
    z range: [0.00380146, 0.99976891]
    """
    xMin = np.min(data["x"].to_numpy())
    xMax = np.max(data["x"].to_numpy())

    yMin = np.min(data["y"].to_numpy())
    yMax = np.max(data["y"].to_numpy())

    zMin = np.min(data["z"].to_numpy())
    zMax = np.max(data["z"].to_numpy())

    out = data.to_numpy(copy=True)
    out[:, 0] = (out[:, 0] - xMin) / (xMax - xMin)
    out[:, 1] = (out[:, 1] - yMin) / (yMax - yMin)
    out[:, 2] = (out[:, 2] - zMin) / (zMax - zMin)

    return out

def undoLinearScaling(data: ndarray, predictions: ndarray) -> ndarray:
    """
    Un-scales predictions from range [0, 1) to the minimum and maximum bounds of each column of the data.
    This method expects two ndarrays of the same shape (samples, 3). Precision is limited to what numpy
    defaults values to (float64)
    """
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    xMin = np.min(x)
    xMax = np.max(x)

    yMin = np.min(y)
    yMax = np.max(y)

    zMin = np.min(z)
    zMax = np.max(z)

    out = np.zeros(data.shape)
    out[:, 0] = predictions[:, 0] * (xMax - xMin) + xMin
    out[:, 1] = predictions[:, 1] * (yMax - yMin) + yMin
    out[:, 2] = predictions[:, 2] * (zMax - zMin) + zMin

    return out

#Ingestion
filename = "training_data.csv"
dataDF = pd.read_csv(filename) #Loading data from csv and loading into a numpy matrix
shuffledDF = dataDF.sample(frac=1, random_state=1).reset_index(drop=True)
datasetSize = 10000
samples     = 5000 #Number of rows to extract for training and testing

trainDF = shuffledDF[:samples]
testDF  = shuffledDF[samples:]

scaledTrainData = linearScaling(trainDF)
scaledTestData  = linearScaling(testDF)

trainInputs  = scaledTrainData[:, [0, 1]]
trainOutputs = scaledTrainData[:, 2].reshape(samples, 1)
testInputs   = scaledTestData[:, [0, 1]]
testOutputs  = scaledTestData[:, 2].reshape(datasetSize - samples, 1)

#Model init args
feat    = 2
batch   = 32
layers  = 2
neurons = [1, 1]
acts:   list[ActivationFunction]
acts    = [ReLU(), Identity()]

#Training args
lr      = 0.1
epochs  = 2
err     = MeanSquaredError()
displayTrain = True
displayTest  = False
saving       = False

#Initializing ANN with args and training on normalized data with training args
model = ANN(numFeatures=feat, batchSize=batch, numLayers=layers, numNeurons=neurons, activations=acts)
model.train(input=trainInputs, z=trainOutputs, learnRate=lr, epochs=epochs, lossFunc=err, displayOutputs=displayTrain)

#Merging training results with input data
predictions = model.test(testInput=trainInputs, displayPredictions=displayTrain)
x = trainInputs[:, 0].reshape(samples, 1) #Convert array of shape (samples,) to matrix of shape (samples, 1)
y = trainInputs[:, 1].reshape(samples, 1)
xy       = np.concatenate((x, y), axis=1) #Concatenate by appending columns (left to right)
xyz      = np.concatenate((xy, predictions), axis=1).reshape(samples, 3)

unscaled = undoLinearScaling(data=trainDF.to_numpy(), predictions=xyz)
#Save training results as csv
if saving:
    saveFilename = "results.csv"
    predsDF = pd.DataFrame(unscaled)
    predsDF = predsDF.rename(columns={0: "x", 1: "y", 2: "z"})
    pd.DataFrame.to_csv(predsDF, saveFilename, index=False)

#Testing
# predictions = model.test(testInput=testInputs, displayPredictions=displayTest)
# print(f"Predicted: {predictions}")
# print(f"Actual: {testOutputs}")