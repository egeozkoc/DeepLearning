from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import numpy as np

class RNN(BaseLayer):
    """
    Implements a Recurrent Neural Network (RNN) layer.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN layer.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Number of output features.
        """
        super().__init__()  # Call base class constructor

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable = True
        self.memorize = False

        # Define Fully Connected Layers Before Setting Weights
        self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc2 = FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        
        # Now it's safe to set weights after defining fc1
        self._weights = self.fc1.weights
        self.optimizer = None

    def forward(self, input_tensor):
        """
        Performs the forward pass through the RNN.

        Args:
            input_tensor (numpy.ndarray): Input tensor of shape (sequence_length, input_size).

        Returns:
            numpy.ndarray: Output tensor of shape (sequence_length, output_size).
        """
        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        
        for i in range(input_tensor.shape[0]):
            input_tensor_tilda = np.concatenate((input_tensor[i], self.hidden_state)).reshape(1, -1)
            self.hidden_state = self.tanh.forward(self.fc1.forward(input_tensor_tilda))
            output_tensor[i] = self.sigmoid.forward(self.fc2.forward(self.hidden_state))
            self.hidden_state = self.hidden_state.ravel()
        
        return output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass through the RNN.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Gradient of the input tensor.
        """
        error_output = np.zeros((error_tensor.shape[0], self.input_size))
        self.gradient_hidden_state = 0
        self.gradient_weights = 0
        
        for i in range(error_tensor.shape[0] - 1, -1, -1):
            backward_step = self.sigmoid.backward(error_tensor[i])
            backward_step = self.fc2.backward(backward_step)
            backward_step += self.gradient_hidden_state
            backward_step = self.tanh.backward(backward_step)
            backward_step = self.fc1.backward(backward_step)
            error_output[i] = backward_step[:, :self.input_size]
            self.gradient_hidden_state = backward_step[:, self.input_size:]
            self.gradient_weights += self.fc1.gradient_weights
        
        if self.optimizer is not None:
            self.set_weights(self.optimizer.calculate_update(self.get_weights(), self.gradient_weights))
        
        return error_output

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initializes weights and biases using the given initializers.

        Args:
            weights_initializer: Initializer for weights.
            bias_initializer: Initializer for biases.
        """
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)
        self._weights = self.fc1.weights

    def get_weights(self):
        """Returns the current weights of the RNN."""
        return self.fc1.weights

    def set_weights(self, weights):
        """Sets new weights for the RNN."""
        self.fc1.weights = weights
        self._weights = self.fc1.weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self.fc1.weights = weights
        self._weights = weights

    def calculate_regularization_loss(self):
        """Computes the regularization loss if a regularizer is used."""
        if self.optimizer and hasattr(self.optimizer, "regularizer"):
            return self.optimizer.regularizer.norm(self._weights)
        return 0
