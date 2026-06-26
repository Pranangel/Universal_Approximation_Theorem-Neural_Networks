from sklearn.preprocessing import MinMaxScaler
import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
import numpy as np
import pandas as pd
import json
import time
import os

#ingest
data = pd.read_csv(os.path.join(os.getcwd(), "data", f"training_data_1.csv")).to_numpy()
size = 10000
x = data[:, 0].reshape(size, 1)
y = data[:, 1].reshape(size, 1)
z = data[:, 2].reshape(size, 1)

#transform
xScaler = MinMaxScaler()
yScaler = MinMaxScaler()
zScaler = MinMaxScaler()

x = xScaler.fit_transform(x)
y = yScaler.fit_transform(y)
z = zScaler.fit_transform(z)
inputs = np.concatenate((x, y), axis=1) #(10000, 2)

#init 1st model w/ Leaky ReLU (100 fan out) -> Leaky ReLU (100 fan out) -> Linear layer (1 fan out)
# MSE loss and standard gradient descent w/ momentum optimization
keras.utils.set_random_seed(37)
m1 = Sequential()
m1.add(Dense(units=100, activation="leaky_relu", kernel_initializer="he_uniform"))
m1.add(Dense(units=100, activation="leaky_relu", kernel_initializer="he_uniform"))
m1.add(Dense(units=1))
m1.compile(loss="mse", optimizer=keras.optimizers.SGD(momentum=0.01))

#init 2nd model w/ ReLU (100 fan out) -> ReLU (100 fan out) -> Linear layer (1 fan out)
# MSE loss and standard gradient descent w/ momentum optimization
m2 = Sequential()
m2.add(Dense(units=100, activation="relu", kernel_initializer="he_uniform"))
m2.add(Dense(units=100, activation="relu", kernel_initializer="he_uniform"))
m2.add(Dense(units=1))
m2.compile(loss="mse", optimizer=keras.optimizers.SGD(momentum=0.01))

tests = [(m1, "keras_leaky"), (m2, "keras_standard_relu")]
for model, name in tests:
    #train and save results
    start = time.time()
    training = model.fit(inputs, z, epochs=20, batch_size=10)
    end = time.time()
    print(f"Training time: {end - start}")

    #Saving losses
    history = training.history 
    with open(os.path.join(os.getcwd(), "training_results", "tests_6", f"{name}_history.json"), 'w') as f:
        json.dump(history, f)
        f.close()

    #test
    predicted = model.predict(inputs)
    xUnscaled = xScaler.inverse_transform(x)
    yUnscaled = yScaler.inverse_transform(y)
    zUnscaled = zScaler.inverse_transform(z)
    zHatUnscaled = zScaler.inverse_transform(predicted)

    print(f"Predicted: {predicted}")
    print(f"MSE: {mean_squared_error(zUnscaled, zHatUnscaled)}")

    #Saving predictions
    results = np.concatenate((xUnscaled, yUnscaled), axis=1)
    results = np.concatenate((results, zHatUnscaled), axis=1)
    saveFilename = os.path.join(os.getcwd(), "training_results", "tests_6", f"{name}_results.csv")
    predsDF = pd.DataFrame(results)
    predsDF = predsDF.rename(columns={0: "x", 1: "y", 2: "z_predicted"})
    pd.DataFrame.to_csv(predsDF, saveFilename, index=False)
