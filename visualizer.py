import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.tri import Triangulation

def visualize(x: ndarray, y: ndarray, z: ndarray, zHat: ndarray,
              epochs: int, trueFunction: str, model: str, savePath: str, display: bool):
    error = zHat - z

    fig = plt.figure(figsize=(16, 10))

    # Create triangulation for irregularly ordered points
    tri = Triangulation(x, y)

    #Predicted surface
    ax1 = fig.add_subplot(221, projection="3d")
    surf1 = ax1.plot_trisurf(tri, zHat, cmap="viridis")
    ax1.set_title(f"Predictions ({epochs} training epochs)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    #True surface
    ax2 = fig.add_subplot(222, projection="3d")
    surf2 = ax2.plot_trisurf(tri, z, cmap="viridis")
    ax2.set_title(f"True Function: {trueFunction}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    #Error heatmap
    ax3 = fig.add_subplot(223)
    tpc = ax3.tripcolor(tri, error, shading="gouraud", cmap="coolwarm")
    fig.colorbar(tpc, ax=ax3)
    ax3.set_title("Error (Predicted - True)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    #Scatterplot of Predicted vs True
    ax4 = fig.add_subplot(224)
    ax4.scatter(z, zHat, alpha=0.4, s=10)

    min_val = min(z.min(), zHat.min())
    max_val = max(z.max(), zHat.max())

    ax4.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Ideal"
    )

    ax4.set_xlabel("True z")
    ax4.set_ylabel("Predicted z")
    ax4.set_title("Predicted vs True")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(savePath, f"{model} Results.png"))
    if (display):
        plt.show()

if __name__ == "__main__":
    tests = [(20, "leaky_model", r"$e^-{\dfrac{(x^2 + y^2)} {\pi}}$"),
            (20, "standard_relu_model", r"$e^-{\dfrac{(x^2 + y^2)} {\pi}}$"),
            (20, "keras_leaky", r"$e^-{\dfrac{(x^2 + y^2)} {\pi}}$"),
            (20, "keras_standard_relu", r"$e^-{\dfrac{(x^2 + y^2)} {\pi}}$"),]

    for epochs, name, functionName in tests:
        #Load data. These strings/paths can be changed per preference
        loadFolder = os.path.join("training_results", "tests_6") #Path to where model outputs (.csv) are saved
        filename   = f"{name}_results.csv" #The name of each .csv
        results    = pd.read_csv(os.path.join(os.getcwd(), loadFolder, filename))
        saveFolder = os.path.join(os.getcwd(), loadFolder)

        x         = results["x"].to_numpy()
        y         = results["y"].to_numpy()
        zHat      = results["z_predicted"].to_numpy()
        z         = np.exp(-(x**2 + y**2) / np.pi)

        visualize(x, y, z, zHat, epochs, functionName, name, saveFolder, False)

    print("Done!")