from Layers.Base import BaseLayer
import numpy as np

def mapReLU(input):
    """
    Applies the ReLU function element-wise.

    Args:
        input (float): Input value.

    Returns:
        float: Output after applying ReLU.
    """
    return input if input > 0 else 0

class ReLU(BaseLayer):
    """
    Implements the Rectified Linear Unit (ReLU) activation function as a layer.
    """
    
    def __init__(self):
        """
        Initializes the ReLU layer.
        """
        super().__init__()
    
    def forward(self, input_tensor):
        """
        Performs the forward pass using the ReLU activation function.

        Args:
            input_tensor (numpy.ndarray): Input tensor.

        Returns:
            numpy.ndarray: Output tensor after applying ReLU.
        """
        self.input_tensor = input_tensor
        vReLU = np.vectorize(mapReLU)
        return vReLU(input_tensor)

    def backward(self, error_tensor):
        """
        Performs the backward pass, computing the gradient of the loss with respect to the input.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input.
        """
        output_error_tensor = np.where(self.input_tensor > 0, 1, 0)
        return np.multiply(output_error_tensor, error_tensor)