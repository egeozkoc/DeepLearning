from Layers.Base import BaseLayer
import numpy as np

class TanH(BaseLayer):
    """
    Implements the Hyperbolic Tangent (TanH) activation function as a layer.
    """
    
    def __init__(self):
        """
        Initializes the TanH layer.
        """
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        """
        Performs the forward pass using the TanH activation function.

        Args:
            input_tensor (numpy.ndarray): Input tensor.

        Returns:
            numpy.ndarray: Output tensor after applying the TanH function.
        """
        output_tensor = np.tanh(input_tensor)
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
        derivative = 1 - np.multiply(self.output_tensor, self.output_tensor)
        output_error = error_tensor * derivative
        return output_error

    def get_output(self):
        """
        Returns the stored output tensor.

        Returns:
            numpy.ndarray: Output tensor after the TanH activation.
        """
        return self.output_tensor

    def set_output(self, output_tensor):
        """
        Sets the output tensor manually.

        Args:
            output_tensor (numpy.ndarray): Precomputed output tensor.
        """
        self.output_tensor = output_tensor
