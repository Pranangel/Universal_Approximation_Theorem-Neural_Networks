# Overview
This is the honors project I completed for my Calculus III class, meant to study the applications of neural networks as universal approximators. (Really, it was more of a deep dive into machine learning by studying the intersection of computer science and math. Thanks, Professor Killebrew! :smile:)

# Research & Development Process
1. Basic literature review of activations, forward and backward propagation
2. Developed and wrote 2-layer architecture
3. Tested forward propagation implementation
4. Modularized layers in a Layer class
5. Stored activations in a global dictionary
6. Implemented backwards propagation manually (after a lot of debugging). Returned to literature to review terminology and assist with debugging.
7. Modularized activations, losses, and initializers

# Explanation of Modules
- [model.py](model.py): The main driver behind the project. Class ANN keeps a dynamic array of Layer classes. This dynamic array acts as a flow graph for forward and backward propagation.
- [activations.py](activations.py): Houses all activation functions (with derivatives), used in model.py
- [initializers.py](initializers.py): Houses all functions for weight initialization, used in model.py
- [losses.py](losses.py): Houses all loss functions (with derivatives) for back propagation. Used in model.py.
- [data_generator.py](data_generator.py): Generates datasets from parametric functions.
- [custom_model_tester.py](custom_model_tester.py) and [keras_model_tester.py](keras_model_tester.py): The files I ran tests on for my custom model and Keras, respectively.

# Architecture Planning
The pictures below show the math-heavy side of what I originally planned for my model: 1 hidden Sigmoid layer and 1 Sigmoid output. This section includes notation, forward and backward propagation (including matrix dimension alignment), and a more faithful version of my current architecture, scaled from my initial designs.

## Notation
<img width="1610" height="916" alt="image" src="https://github.com/user-attachments/assets/c8acef6c-e3f0-40b3-bc49-b169578b5c99" />

## Forward Propagation
<img width="2084" height="702" alt="image" src="https://github.com/user-attachments/assets/63fd844c-addc-41e9-851e-2b3257a201fd" />

## Backward Propagation
<img width="1096" height="1376" alt="image" src="https://github.com/user-attachments/assets/653cf563-28d1-41e0-9d72-582f79357e95" />
The top left directed graph is what I started with; then, below it, I sorted out the individual partials; then I pieced them together (the Math curly brace); and then I went into deeper detail, rearranging the math to match up loosely to code (Code curly brace). There are a lot of colors at play here, but the most important ones I want to highlight are the teal and dark purple partials. They essentially reflect a shared piece of the gradient that works its way towards the first hidden layer, which helped me implement the code version of back propagation. (The teal color is the output layer's gradient, the purple is the 1st hidden layer's gradient.)

## And Scaling it a Little Further...
<img width="1512" height="924" alt="image" src="https://github.com/user-attachments/assets/5e8382e7-b99a-4d21-87aa-34baba9270b1" />
This is the design most similar to the architecture I used in my tests! The only difference is the output layer is a ReLU (notated by the R function), and there are a LOT more colors (again, to piece together how gradients worked up the graph in code).

# Testing Methodology
I compared my implementation with Keras's Sequential model, measuring training runtimes and loss (mean squared error) on the Gaussian function $e^{-({x^2}+{y^2})}$. The models had the same parameters and architecture (see below). In a seperate test, I kept the parameters and architecture the same but replaced the activations with Leaky ReLU to compare how they performed with their ReLU counterparts. To visualize, I tested the models on the same data after training.

Model Training Parameters (custom and Keras):
- 20 epochs
- Batch size = 10
- Learn rate = 0.01
- Uniform He weight initialization
- Bias 0 initialization
- Mean squared error per sample loss
- Mini-batch gradient descent with momentum optimization (constant of 0.01)

Model Architecture (custom and Keras):
- 1st Layer: 100 neuron (fan out) with ReLU
- 2nd Layer: 100 neuron (fan out) with ReLU
- Output Layer: 1 neuron (fan out) linear

# Results
|   | My Model | Keras |
| ------------- | ------------- | ------------- |
| Normal ReLU | <img width="1600" height="1000" alt="standard_relu_model Results" src="https://github.com/user-attachments/assets/8801eb4d-eb69-4116-b81d-d2a7d6687e3e" /> Epoch 1 Loss: 0.0464<br>Epoch 20 loss: 0.0008<br>Training time: 4.9222 seconds | <img width="1600" height="1000" alt="keras_standard_relu Results" src="https://github.com/user-attachments/assets/922e8162-d327-4435-9c73-be0c362229f4" /> Epoch 1 Loss: 0.0560<br>Epoch 20 Loss: 0.0022<br>Training time: 24.6033 seconds |
| Leaky ReLU | <img width="1600" height="1000" alt="leaky_model Results" src="https://github.com/user-attachments/assets/6eb8f9dc-e75f-4638-ab84-e1602ee58fbc" /> Epoch 1 Loss: 0.0497<br>Epoch 20 loss: 0.0043<br>Training time: 5.8210 seconds | <img width="1600" height="1000" alt="keras_leaky Results" src="https://github.com/user-attachments/assets/c9398899-9942-4cc7-8308-e8e747d6853a" /> Epoch 1 Loss: 0.0593<br>Epoch 20 Loss: 0.0025<br>Training time: 25.0169 seconds |

# Discussion
My custom model ran faster in both tests, likely because the only overhead was Numpy's operations being applied to the matrices. (In terms of scalability, however, my model does not support GPU optimization, nor does it handle out-of-memory datasets.) In the ReLU test, my model achieved a lower loss than the Keras model after the 20th epoch; however, in the Leaky ReLU test, Keras achieved a lower loss, possibly due to more favorable random initialization.

# Challenges
- Backwards Propagation: I originally began with a 1-layer design and pivoted to a 2-layer after encountering issues with backpropagation. I did not understand the difference between a naive matrix product and matrix multiplication until I studied deeper. I fixed the algorithm to only apply matrix multiplication when taking the partial derivative of an activated output w.r.t. the un-activated output of a layer.
- Data ingestion: When I began my 2-layer design, I was not accessing the correct columns of my data, causing my model to train on faulty data.
- Model training: I had a critical error in shuffling during batch training causing the model to take the x and z columns as inputs instead of the x and y. Fixing this and adding momentum optimization dramatically improved results.

# Future Improvements!
- Add support for convolutional neural networks
- Add save/load functionality
- Add a logger or something that tracks training history
- Add different optimizers for parameter updates
