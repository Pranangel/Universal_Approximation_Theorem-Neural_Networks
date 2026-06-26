#Author: Pranangel
#Purpose: File to train model and save predictions.

import time
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from numpy import ndarray
import os

from model import ANN
from activations import *
from losses import *

def ingest(path: str):
    dataDF = pd.read_csv(path)

    #Shuffling Pandas DataFrame and preparing training DataFrame
    # rows, cols = dataDF.shape
    shuffledDF = dataDF.sample(frac=1, random_state=1).reset_index(drop=True)

    return shuffledDF

def linearScaling(data: DataFrame) -> ndarray:
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
    This method expects two ndarrays of the same shape (samples, 3). Precision is limited numpy defaults
    (float64). The data matrix is needed to extract minimums and maximums for each column, applied to the
    predictions matrix to scale the predictions back to normal.
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

def savePredictions(preds: ndarray, saveDst: str):
    predsDF = pd.DataFrame(preds)
    predsDF = predsDF.rename(columns={0: "x", 1: "y", 2: "z_predicted"})
    pd.DataFrame.to_csv(predsDF, saveDst, index=False)

def runTests(tests: list, batchSize: int, dataDF: DataFrame, trainInputs: ndarray, trainOutputs: ndarray, predicting: bool, savePath=None):
    i = 1
    timeDiff = 0.0
    for model, args, modelName in tests:
        #Logging training time for each model
        start = time.time()
        losses = model.train(
            input         =trainInputs,
            expected      =trainOutputs,
            batchSize     =batchSize,
            learnRate     =args["learn_rate"],
            epochs        =args["epochs"],
            lossFuncName  =args["error"],
            displayOutputs=False
        )
        end = time.time()
        timeDiff = end - start

        #Printing losses and runtime
        print(f"{modelName} Results:\nLosses:")
        if (len(losses) >= 10):
            end = len(losses) - 1
            print(f"[{losses[0]}, {losses[1]}, {losses[2]}, {losses[3]}, {losses[4]}, ..., {losses[end - 4]}, {losses[end - 3]}, {losses[end - 2]}, {losses[end - 1]}, {losses[end]}]")
        else:
            print(losses)
        print(f"Training runtime: {timeDiff} seconds")

        if (predicting):
            print("Running model on prediction tasks:")

            #Merging training results with input data
            predictions = model.test(testInput=trainInputs, actual=trainOutputs, lossFuncName="mse")
            concatPredictions = np.concatenate((trainInputs, predictions), axis=1).reshape(samples, 3)
            unscaledPredictions = undoLinearScaling(data=dataDF.to_numpy(), predictions=concatPredictions)

            print(f"Predicted: {predictions}")

            if (savePath != None):
                print("Saving results...")
                
                #Save training results as csv
                savePredictions(unscaledPredictions, os.path.join(savePath, f"{modelName}_results.csv"))
        i += 1

    print("Done\n")

files = [os.path.join(os.getcwd(), "data", "training_data_1.csv")]
savePath = os.path.join(os.getcwd(), "training_results", "tests_6")

#Setting initialization and training parameters
trainingArgs = {
    "learn_rate": 0.01,
    "epochs": 20,
    "error": "mse",
}

model1 = ANN(inputFeatures=2)
model1.add(output_features=100, activation="leaky_relu")
model1.add(output_features=100, activation="leaky_relu")
model1.add(output_features=1, activation="identity")
model1Name = "leaky_model"

model2 = ANN(inputFeatures=2)
model2.add(output_features=100, activation="relu")
model2.add(output_features=100, activation="relu")
model2.add(output_features=1, activation="identity")
model2Name = "standard_relu_model"

tests = [(model1, trainingArgs, model1Name), (model2, trainingArgs, model2Name)]

for file in files:
    #Ingestion
    shuffledDF = ingest(file)

    #Scaling data
    inputFeatures = 2
    data = linearScaling(shuffledDF)
    samples, features = data.shape
    trainInputs  = data[:, :inputFeatures].reshape(samples, inputFeatures)
    trainOutputs = data[:, inputFeatures:].reshape(samples, features - inputFeatures)

    runTests(tests=tests,
             batchSize=10,
             dataDF=shuffledDF,
             trainInputs=trainInputs,
             trainOutputs=trainOutputs,
             predicting=False, #Set me to True if you want to save results
             savePath=savePath
    )
