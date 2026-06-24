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
- [trainer.py](trainer.py): The main file I ran tests on.

# Notation
TODO

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
| Normal ReLU | <img width="1600" height="1000" alt="standard_relu_model Results" src="https://github.com/user-attachments/assets/8801eb4d-eb69-4116-b81d-d2a7d6687e3e" /> Epoch 1 Loss: 0.04635847223715623<br>Epoch 20 loss: 0.0008274814189021644<br>Training time: 4.922151565551758 seconds | <img width="1600" height="1000" alt="keras_standard_relu Results" src="https://github.com/user-attachments/assets/922e8162-d327-4435-9c73-be0c362229f4" /> Epoch 1 Loss: 0.0560<br>Epoch 20 Loss: 0.0022<br>Training time: 24.603302240371704 seconds |
| Leaky ReLU | <img width="1600" height="1000" alt="leaky_model Results" src="https://github.com/user-attachments/assets/6eb8f9dc-e75f-4638-ab84-e1602ee58fbc" /> Epoch 1 Loss: 0.049720800754544645<br>Epoch 20 loss: 0.00427488328512963<br>Training time: 5.820997953414917 seconds | <img width="1600" height="1000" alt="keras_leaky Results" src="https://github.com/user-attachments/assets/c9398899-9942-4cc7-8308-e8e747d6853a" /> Epoch 1 Loss: 0.0593<br>Epoch 20 Loss: 0.0025<br>Training time: 25.0168936252594 seconds |

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
