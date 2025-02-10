# Deep Learning Framework from Scratch

## Overview
This repository contains a deep learning framework implemented from scratch in Python. It includes core neural network layers, optimization techniques, regularization methods, and recurrent layers, providing a comprehensive understanding of deep learning fundamentals.

## Features
- **Neural Network Layers**: Fully Connected, Convolutional, Pooling, Dropout, Batch Normalization, and Activation Functions (ReLU, Sigmoid, Tanh)
- **Optimization Algorithms**: Stochastic Gradient Descent (SGD), SGD with Momentum, Adam Optimizer
- **Regularization Techniques**: L1/L2 Regularization, Dropout, Batch Normalization
- **Recurrent Layers**: Elman RNN, LSTM (Optional)
- **Loss Functions**: Cross-Entropy Loss
- **Testing Framework**: Unit tests to validate implementations

## Repository Structure
DEEPLEARNING/ │── Layers/ # Neural network layers implementation │ │── init.py │ │── Base.py # Base class for layers │ │── FullyConnected.py # Fully connected layer │ │── Conv.py # Convolutional layer │ │── Pooling.py # Pooling layer │ │── Dropout.py # Dropout layer │ │── BatchNormalization.py # Batch normalization layer │ │── Flatten.py # Flatten layer │ │── RNN.py # Recurrent layer (Elman RNN) │ │── Sigmoid.py # Sigmoid activation function │ │── TanH.py # Tanh activation function │ │── Initializers.py # Weight initialization methods │ │── Helpers.py # Utility functions │── Optimization/ # Optimization and regularization methods │ │── init.py │ │── Constraints.py # L1/L2 regularization │ │── Loss.py # Loss functions (Cross-Entropy) │ │── Optimizers.py # Optimizers (SGD, Adam, Momentum) │── .gitattributes # Git configuration

