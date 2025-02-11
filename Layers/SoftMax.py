import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    """
    Implements the SoftMax activation function as a layer.
    """
    
    def __init__(self):
        """
        Initializes the SoftMax layer.
        """
        super().__init__()

    def forward(self, input_tensor):
        """
        Performs the forward pass using the SoftMax activation function.

        Args:
            input_tensor (numpy.ndarray): Input tensor.

        Returns:
            numpy.ndarray: Output tensor after applying SoftMax.
        """
        self.input_tensor = input_tensor
        max_x = np.max(input_tensor, axis=1, keepdims=True)
        input_tensor = input_tensor - max_x
        exp_input_tensor = np.exp(input_tensor)
        sum_batch = np.sum(exp_input_tensor, axis=1, keepdims=True)
        output_tensor = exp_input_tensor / sum_batch
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass, computing the gradient of the loss with respect to the input.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input.
        """
        previous_error_tensor = np.multiply(self.output_tensor, (error_tensor - np.sum(np.multiply(error_tensor, self.output_tensor), axis=1, keepdims=True)))
        return previous_error_tensor