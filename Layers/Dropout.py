from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    """
    Implements the Dropout layer, a regularization technique to prevent overfitting
    by randomly deactivating neurons during training.
    """
    
    def __init__(self, probability):
        """
        Initializes the Dropout layer.

        Args:
            probability (float): The probability of keeping a neuron active during training.
        """
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        """
        Performs the forward pass of the Dropout layer.

        Args:
            input_tensor (numpy.ndarray): Input tensor.

        Returns:
            numpy.ndarray: Output tensor after applying dropout.
        """
        if self.testing_phase == False:
            random_numbers = np.random.uniform(0, 1, input_tensor.shape)
            dropout_matrix = random_numbers < self.probability
            output_tensor = np.multiply(dropout_matrix, input_tensor)
            output_tensor = np.multiply(output_tensor, 1/self.probability)
            self.dropout_matrix = dropout_matrix
            return output_tensor
        else:
            return input_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass of the Dropout layer.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Gradient of the input tensor.
        """
        if self.testing_phase == False:
            output_error = np.multiply(self.dropout_matrix, error_tensor)
            output_error = np.multiply(output_error, 1 / self.probability)
            return output_error
