# Overview
This is an honors project studying the applications of neural networks as universal approximators for my Calculus III class. I developed and trained a deep learning from scratch on a dataset of 10,000 points, comparing results to a Keras model with the same parameters and hyperparameters. I want to thank Professor Killebrew at Mesa Community College for learning with, teaching, and supporting me! :smile:

# Research & Development Process
1. Basic literature review of activations, forward and backward propagation
2. Developed and wrote 2-layer architecture
3. Tested forward propagation implementation
4. Modularized layers in a Layer class
5. Stored activations in a global dictionary
6. Implemented backwards propagation manually (after a lot of debugging); returned to literature to review terminology and assist with debugging
7. Modularized activations, losses, and initializers

# Explanation of Modules and Files
- [model.py](model.py): The main driver behind the project. Class ANN keeps a dynamic array of Layer classes. This dynamic array acts as a flow graph for forward and backward propagation.
- [activations.py](activations.py): Houses all activation functions (with derivatives), used in model.py
- [initializers.py](initializers.py): Houses all functions for weight initialization, used in model.py
- [losses.py](losses.py): Houses all loss functions (with derivatives) for back propagation. Used in model.py.
- [data_generator.py](data_generator.py): Generates datasets from parametric functions.
- [custom_model_tester.py](custom_model_tester.py) and [keras_model_tester.py](keras_model_tester.py): The files I ran tests on for my custom model and Keras, respectively. Both files will output a .csv file containing x, y, and z_predicted columns. These can be used in visualizer.py (see below).
- [visualizer.py](visualizer.py): The file I used to generate visualizations. It uses the .csv files generated from the above tester files to produce a residual heatmap, predicted surface, actual surface, and a scatterplot comparing the predicted z-values to the actual z-values.
- [training_data.csv](training_data.csv) and [training_data_1.csv](training_data_1.csv): Files containing 10,000 points on the Gaussian function $f(x,y)$ = $exp(-{{(x^2 + y^2)} / \pi}$). Prof. Killebrew provided me with training_data.csv during my initial development, and I generated training_data_1.csv from data_generator.py. (Either file can be used, but note that training_data.csv is pre-shuffled.)

# Architecture Planning
The pictures below show the math-heavy side of what I originally planned for my model: 1 hidden Sigmoid layer and 1 Sigmoid output. This section includes notation, forward and backward propagation (including matrix dimension alignment), and a more faithful version of my current architecture, scaled from my initial designs.

## Notation
<img width="1818" height="952" alt="image" src="https://github.com/user-attachments/assets/0476a5e2-9f43-4c8b-9f67-a5a6780aa347" />

## Forward Propagation
<img width="2084" height="702" alt="image" src="https://github.com/user-attachments/assets/63fd844c-addc-41e9-851e-2b3257a201fd" />

## Backward Propagation
<img width="1096" height="1376" alt="image" src="https://github.com/user-attachments/assets/653cf563-28d1-41e0-9d72-582f79357e95" />
The top left directed graph is what I started with; then, below it, I sorted out the individual partials; then I pieced them together (the Math curly brace); and then I went into deeper detail, rearranging the math to match up loosely to code (Code curly brace). There are a lot of colors at play here, but the most important ones I want to highlight are the teal and dark purple partials. They essentially reflect a shared piece of the gradient that works its way towards the first hidden layer, which helped me implement the code version of back propagation. (The teal color is the output layer's gradient, the purple is the 1st hidden layer's gradient.)

## And Scaling it a Little Further...
<img width="1512" height="924" alt="image" src="https://github.com/user-attachments/assets/5e8382e7-b99a-4d21-87aa-34baba9270b1" />
This is the design most similar to the architecture I used in my tests! The only difference is the output layer is a ReLU (notated by the R function), and there are a LOT more colors (again, to piece together how gradients worked up the graph in code).

# Testing Methodology
I compared my implementation with Keras's Sequential model, measuring training runtimes and loss (mean squared error) on the Gaussian function $f(x,y)$ = $exp(-{{(x^2 + y^2)} / \pi}$). The models had the same parameters and architecture (see below). In a seperate test, I kept the parameters and architecture the same but replaced the activations with Leaky ReLU to compare how they performed with their ReLU counterparts. To visualize, I tested the models on the same data after training.

**_(Note: I chose a Leaky ReLU constant of 0.003 for my model; Keras uses 0.3 by default, which I did not change.)_**

## Model Training Parameters (custom and Keras):
- 20 epochs
- Batch size = 10
- Learn rate = 0.01
- Uniform He weight initialization
- Bias 0 initialization
- Mean squared error per sample loss
- Mini-batch gradient descent with momentum optimization (constant of 0.01)

## Model Architecture (custom and Keras):
- 1st Layer: 100 neuron (fan out) with ReLU
- 2nd Layer: 100 neuron (fan out) with ReLU
- Output Layer: 1 neuron (fan out) linear

# Results
|   | My Model | Keras |
| ------------- | ------------- | ------------- |
| Normal ReLU | <img width="1600" height="1000" alt="standard_relu_model Results" src="https://github.com/user-attachments/assets/49ef2502-ee2d-44cd-bbc9-2f89ba6727da" /> Epoch 1 Loss: 0.0464<br>Epoch 20 loss: 0.0008<br>Training time: 4.9222 seconds | <img width="1600" height="1000" alt="keras_standard_relu Results" src="https://github.com/user-attachments/assets/83d4b722-e67f-4147-9897-0c3cefbe194e" /> Epoch 1 Loss: 0.0560<br>Epoch 20 Loss: 0.0022<br>Training time: 24.6033 seconds |
| Leaky ReLU | <img width="1600" height="1000" alt="leaky_model Results" src="https://github.com/user-attachments/assets/69783e85-a442-457a-8205-982e5054b56c" /> Epoch 1 Loss: 0.0497<br>Epoch 20 loss: 0.0043<br>Training time: 5.8210 seconds | <img width="1600" height="1000" alt="keras_leaky Results" src="https://github.com/user-attachments/assets/e4890057-b0de-42c9-80d6-bd3f28606487" /> Epoch 1 Loss: 0.0593<br>Epoch 20 Loss: 0.0025<br>Training time: 25.0169 seconds |

# Discussion
My custom model ran faster in both tests, likely because the only overhead was Numpy's operations being applied to the matrices. (In terms of scalability, however, my model does not support GPU optimization, nor does it handle out-of-memory datasets.) In the ReLU test, my model achieved a lower loss than the Keras model after the 20th epoch; however, in the Leaky ReLU test, Keras achieved a lower loss, possibly due to Keras using a Leaky ReLU constant of 0.3 compared to my model's constant of 0.003. This means the Leaky ReLU gradients in Keras are greater by a magnitude of 100, speeding up gradient updates in the process.

I was surprised to observe visually that Keras's predicted surface and error heatmap looked very similar for both the regular and Leaky ReLU models. **What's happening under the hood that causes Keras to produce similar results regardless of ReLU or Leaky ReLU?**

I was just as surprised to find that my model using normal ReLU had a very close predicted surface to the actual (and a more neutral heatmap), which leads me to ask: **What is the key factor or factors causing my model's normal ReLU to outperform Keras's normal ReLU in terms of accuracy?**

# Challenges
- Backwards Propagation: I originally began with a 1-layer design and pivoted to a 2-layer after encountering issues with backpropagation. I did not understand the difference between a naive matrix product and matrix multiplication until I studied deeper. I fixed the algorithm to only apply matrix multiplication when taking the partial derivative of an activated output w.r.t. the un-activated output of a layer.
- Data ingestion: When I began my 2-layer design, I was not accessing the correct columns of my data, causing my model to train on faulty data.
- Model training: I had a critical error in shuffling during batch training causing the model to take the x and z columns as inputs instead of the x and y. Fixing this and adding momentum optimization dramatically improved results.

# Future Improvements!
- Add support for convolutional neural networks
- Add save/load functionality
- Add a logger or something that tracks training history
- Add different optimizers for parameter updates
