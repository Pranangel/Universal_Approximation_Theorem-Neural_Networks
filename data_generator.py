#Author: Pranangel
#Purpose: Generating data points for different types of curves.

import math
import csv
import os
from typing import Generator

class DataGenerator:

    @staticmethod
    def __spring():
        """
        
        """
        t = 0.
        end = 2 * math.pi
        target = 10_000
        rate = end / target

        while (t < end):
            x = round(math.cos(t), 12)
            y = round(math.sin(t), 12)
            z = round(0.1 * t, 12)
            
            yield (x, y, z)
            t += rate

    @staticmethod
    def __doubleSaddle():
        u = -2.
        v = -2.
        end = 2.

        length = end - u
        targetPoints = 10_000
        rate = length / math.sqrt(targetPoints) #Because the below loop is O(N^2)

        while u < end:
            while v < end:
                v = round(v + rate, 12)
                x = 2 * u
                y = 2 * v
                z = round((((7 * (x) * (y)) / math.exp((x)**2 + (y)**2)) / 3) + .5, 12)
                yield (u,v,z)

            u = round(u + rate, 12)
            v = -end

    @staticmethod
    def __gaussianSombrero():
        u = -3.0
        v = -3.0
        end = 3.0

        length = end - u
        targetPoints = 10_000
        rate = length / math.sqrt(targetPoints) #Because the below loop is O(N^2)

        while (u < end):
            while (v < end):
                v = round(v + rate, 12)
                z = round(math.exp(-(u**2 + v**2) / math.pi), 12)
                yield (u,v,z)

            u = round(u + rate, 12)
            v = -end
    
    datasets = {
        "spring": __spring(),
        "double_saddle": __doubleSaddle(),
        "gaussian": __gaussianSombrero(),
    }

    @staticmethod
    def createDataset(funcName: str, savePath: str, saveFileName=""):
        """
        Note 1: This will always make a new csv named to whatever the saveFileName argument
        is. If saveFileName is an empty string, the csv will be titled in the format
        "training_data_{i}.csv", where 'i' is a number assigned uniquely to each file. (It's
        a basic while loop, which means if there's a file named "training_data_1.csv" and
        "training_data_3.csv", the newly generated file will be called "training_data_2.csv")

        Note 2: Each of the functions which generate the data points internally round
        all calculations to the nearest 12 digits to avoid overflows.
        """

        if (funcName in DataGenerator.datasets.keys()):
            func = DataGenerator.datasets[funcName]

            name = saveFileName
            if (name == ""):
                i = 1
                name = f"training_data_{i}.csv"
                while (os.path.exists(os.path.join(savePath, name))):
                    i += 1
                    name = f"training_data_{i}.csv"

            with open(os.path.join(savePath, name), 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])

                for x, y, z in func:
                    writer.writerow([x, y, z])
                    
                f.close()
        else:
            print(f"Error: key {funcName} not in datasets!")
            raise KeyError

if __name__ == "__main__":
    savePath =  os.path.join(os.getcwd(), "data") #change me
    DataGenerator.createDataset("gaussian", savePath)
    print("Done!")