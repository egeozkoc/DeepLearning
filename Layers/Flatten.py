import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    """
    Implements a Flatten layer, which reshapes multi-dimensional input into a 2D tensor.
    """
    
    def __init__(self):
        """
        Initializes the Flatten layer.
        """
        super().__init__()

    def forward(self, input_tensor):
        """
        Performs the forward pass by flattening the input tensor.

        Args:
            input_tensor (numpy.ndarray): Input tensor of any shape.

        Returns:
            numpy.ndarray: Flattened 2D tensor.
        """
        flattened_tensor = []
        self.original_shape = input_tensor.shape
        for i in range(0, input_tensor.shape[0]):
            flattened_tensor.append(input_tensor[i].flatten())
        input_tensor = np.array(flattened_tensor)
        return input_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass by reshaping the error tensor back to its original shape.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Reshaped error tensor matching the original input shape.
        """
        error_tensor = error_tensor.reshape(self.original_shape)
        return error_tensor
