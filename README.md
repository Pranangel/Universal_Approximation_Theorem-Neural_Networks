# Overview
This is the honors project I completed for my Calculus III class, meant to study the applications of neural networks as universal approximators. (Really, it was more of a deep dive into machine learning by studying the intersection of computer science and math. Thanks, Professor Killebrew! :smile:)

# Research & Development Process
TODO

# Explanation of Modules
- [model.py](model.py): The main driver behind the project. Class ANN keeps a dynamic array of Layer classes. This dynamic array acts as a flow graph for forward and backward propagation.
- [activations.py](activations.py): Houses all activation functions (with derivatives), used in model.py
- [initializers.py](initializers.py): Houses all functions for weight initialization, used in model.py
- [losses.py](losses.py): Houses all loss functions (with derivatives) for back propagation. Used in model.py.
- [data_generator.py](data_generator.py): The file I used to generate the different datasets.
- [trainer.py](trainer.py): The main file I ran tests on.

# Notation
TODO

# Results
TODO

# Challenges
TODO

# Future Improvements!
- Add support for convolutional neural networks
- Add save/load functionality
- Add a logger or something that tracks training history
- Add different optimizers for parameter updates
