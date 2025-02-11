from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    """
    Implements a max pooling layer for a neural network.
    """
    
    def __init__(self, stride_shape, pooling_shape):
        """
        Initializes the Pooling layer.

        Args:
            stride_shape (tuple): The stride size for pooling.
            pooling_shape (tuple): The shape of the pooling filter.
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        """
        Performs the forward pass of the pooling layer.

        Args:
            input_tensor (numpy.ndarray): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            numpy.ndarray: Pooled output tensor.
        """
        self.input_shape = input_tensor.shape
        output_tensor = np.zeros([
            input_tensor.shape[0], input_tensor.shape[1], 
            input_tensor.shape[2] - self.pooling_shape[0] + 1, 
            input_tensor.shape[3] - self.pooling_shape[1] + 1
        ])
        max_position = np.zeros(output_tensor.shape)
        
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                for k in range(output_tensor.shape[2]):
                    for l in range(output_tensor.shape[3]):
                        submatrix = input_tensor[i, j, k:k + self.pooling_shape[0], l:l + self.pooling_shape[1]]
                        output_tensor[i, j, k, l] = np.max(submatrix)
                        max_position[i, j, k, l] = np.argmax(submatrix)

        strided_output_tensor = output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        self.max_position = max_position[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        return strided_output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass of the pooling layer.

        Args:
            error_tensor (numpy.ndarray): Error tensor from the next layer.

        Returns:
            numpy.ndarray: Gradient of the input tensor.
        """
        previous_error_tensor = np.zeros(self.input_shape)
        
        for i in range(error_tensor.shape[0]):
            for j in range(error_tensor.shape[1]):
                for k in range(error_tensor.shape[2]):
                    for l in range(error_tensor.shape[3]):
                        x = int(k * self.stride_shape[0] + self.max_position[i, j, k, l] // self.pooling_shape[1])
                        y = int(l * self.stride_shape[1] + self.max_position[i, j, k, l] % self.pooling_shape[1])
                        previous_error_tensor[i, j, x, y] += error_tensor[i, j, k, l]
        
        return previous_error_tensor
