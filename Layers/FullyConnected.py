from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    """
    Implements a Fully Connected (Dense) layer in a neural network.
    """
    
    def __init__(self, input_size, output_size):
        """
        Initializes the FullyConnected layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.optimizer = None
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

    def forward(self, input_tensor):
        """
        Performs the forward pass of the FullyConnected layer.

        Args:
            input_tensor (numpy.ndarray): Input tensor of shape (batch_size, input_size).

        Returns:
            numpy.ndarray: Output tensor of shape (batch_size, output_size).
        """
        bias = np.ones([input_tensor.shape[0], 1])
        self.input_tensor = np.concatenate([input_tensor, bias], axis=1)
        output_tensor = np.matmul(self.input_tensor, self.weights)
        self.batch_size = output_tensor.shape[0]
        self.output_size = output_tensor.shape[1]
        return output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass, computing gradients for weight updates.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Gradient of the input tensor.
        """
        previous_error_tensor = np.matmul(error_tensor, np.transpose(self.weights))
        previous_error_tensor = np.delete(previous_error_tensor, -1, 1)
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return previous_error_tensor

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer for the layer.

        Args:
            optimizer: The optimizer to use for weight updates.
        """
        self.optimizer = optimizer

    def get_optimizer(self):
        """
        Returns the optimizer used by the layer.

        Returns:
            Optimizer instance or None.
        """
        return self.optimizer

    def set_input(self, input_tensor):
        """
        Sets the input tensor.

        Args:
            input_tensor (numpy.ndarray): Input tensor.
        """
        self.input_tensor = input_tensor

    def get_input(self):
        """
        Returns the stored input tensor.

        Returns:
            numpy.ndarray: Input tensor.
        """
        return self.input_tensor

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initializes weights and bias using the provided initializers.

        Args:
            weights_initializer: Initializer for weights.
            bias_initializer: Initializer for bias.
        """
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.concatenate([self.weights, self.bias], axis=0)

    def calculate_regularization_loss(self):
        """
        Computes the regularization loss if a regularizer is used.

        Returns:
            float: Regularization loss value.
        """
        regularization_loss = self.optimizer.regularizer.norm(self.weights)
        return regularization_loss
