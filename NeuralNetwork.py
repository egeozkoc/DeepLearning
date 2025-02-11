import Layers.FullyConnected
from Optimization.Loss import CrossEntropyLoss
import copy


class NeuralNetwork:
    """
    A class representing a neural network model with multiple layers, loss function,
    and optimization strategies.
    """
    
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        """
        Initializes the NeuralNetwork.

        Args:
            optimizer: The optimization algorithm to use.
            weights_initializer: Function or method to initialize weights.
            bias_initializer: Function or method to initialize biases.
        """
        self.optimizer = copy.deepcopy(optimizer)
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = CrossEntropyLoss()
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def next(self):
        """
        Placeholder for fetching the next batch of data.
        """
        pass

    def forward(self):
        """
        Performs a forward pass through the network, computing the loss.

        Returns:
            float: The total loss including regularization loss.
        """
        input_tensor, self.label_tensor = self.data_layer.next()

        regularization_loss = 0
        for i in range(0, len(self.layers)):
            self.layer = self.layers[i]
            input_tensor = self.layer.forward(input_tensor)
            if isinstance(self.layer, Layers.FullyConnected.FullyConnected) or isinstance(self.layer, Layers.RNN.RNN):
                if hasattr(self.layer, 'optimizer'):
                    if hasattr(self.layer.optimizer, 'regularizer'):
                        regularization_loss += self.layer.calculate_regularization_loss()

        loss = self.loss_layer.forward(input_tensor, self.label_tensor) + regularization_loss
        return loss

    def backward(self):
        """
        Performs a backward pass through the network, propagating the error.

        Returns:
            numpy.ndarray: The propagated error after the backward pass.
        """
        error = self.loss_layer.backward(self.label_tensor)

        for i in range(len(self.layers) - 1, -1, -1):
            self.layer = self.layers[i]
            error = self.layer.backward(error)

        return error

    def append_layer(self, layer):
        """
        Adds a new layer to the network.

        Args:
            layer: The layer to be added.
        """
        if layer.trainable is True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        """
        Trains the neural network for a given number of iterations.

        Args:
            iterations (int): The number of training iterations.
        """
        self.phase(False)
        for i in range(0, iterations):
            self.forward()
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        """
        Tests the neural network by running a forward pass on the input tensor.

        Args:
            input_tensor (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The network output after the forward pass.
        """
        self.phase(True)
        for i in range(0, len(self.layers)):
            self.layer = self.layers[i]
            input_tensor = self.layer.forward(input_tensor)
        return input_tensor

    def phase(self, testing_phase):
        """
        Sets the phase of the network (training or testing).

        Args:
            testing_phase (bool): If True, the network is in testing mode; otherwise, it is in training mode.
        """
        for i in range(0, len(self.layers)):
            self.layers[i].testing_phase = testing_phase
