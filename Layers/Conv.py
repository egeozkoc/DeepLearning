import numpy as np
from scipy.signal import convolve, correlate
from Layers.Base import BaseLayer
import copy

class Conv(BaseLayer):
    """
    Implements a convolutional layer for neural networks.

    This layer performs convolution operations on input tensors using learnable
    filters (kernels). It supports both 2D (image-like) and 1D (sequence-like) inputs.

    Attributes:
        trainable (bool): Indicates whether the layer parameters are trainable.
        stride_shape (tuple): Stride of the convolution operation.
        convolution_shape (tuple): Shape of the convolutional kernels.
        num_kernels (int): Number of kernels (filters) used in the layer.
        weights (numpy.ndarray): Learnable weight parameters (filters).
        bias (numpy.ndarray): Learnable bias parameters.
    """

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        Initializes the Conv layer with given stride, kernel shape, and number of kernels.

        Args:
            stride_shape (tuple): The stride of the convolution operation.
            convolution_shape (tuple): The shape of the convolutional kernels (filters).
            num_kernels (int): The number of kernels used in the layer.
        """
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, (self.num_kernels,) + self.convolution_shape)
        self.bias = np.random.uniform(0, 1, self.num_kernels)

    def forward(self, input_tensor):
        """
        Performs the forward pass of the convolutional layer.

        Applies the convolution operation to the input tensor and performs stride-based downsampling.

        Args:
            input_tensor (numpy.ndarray): Input tensor with shape (batch_size, channels, height, width)
                for 2D data or (batch_size, channels, sequence_length) for 1D data.

        Returns:
            numpy.ndarray: The output tensor after applying convolution and striding.
        """
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 4:
            output_tensor = []
            for i in range(input_tensor.shape[0]):
                output_single_batch = []
                for j in range(self.num_kernels):
                    output_single_conv = np.zeros([input_tensor.shape[2], input_tensor.shape[3]])
                    for k in range(input_tensor.shape[1]):
                        output_single_channel = correlate(input_tensor[i][k], self.weights[j][k], 'same')
                        output_single_conv += output_single_channel
                    output_single_batch.append(output_single_conv + self.bias[j])
                output_tensor.append(output_single_batch)
            output_tensor = np.array(output_tensor)

            if len(self.stride_shape) == 1:
                stride_shape = np.concatenate(self.stride_shape, self.stride_shape)
            elif len(self.stride_shape) == 2:
                stride_shape = self.stride_shape

            strided_output_tensor = np.zeros([output_tensor.shape[0], output_tensor.shape[1], 
                                              ((output_tensor.shape[2] - 1) // stride_shape[0]) + 1, 
                                              ((output_tensor.shape[3] - 1) // stride_shape[1]) + 1])

            for i in range(output_tensor.shape[0]):
                for j in range(output_tensor.shape[1]):
                    y = 0
                    for k in range(0, output_tensor.shape[2], stride_shape[0]):
                        x = 0
                        for l in range(0, output_tensor.shape[3], stride_shape[1]):
                            strided_output_tensor[i][j][y][x] = output_tensor[i][j][k][l]
                            x += 1
                        y += 1

        elif len(input_tensor.shape) == 3:
            output_tensor = []
            for i in range(input_tensor.shape[0]):
                output_single_batch = []
                for j in range(self.num_kernels):
                    output_single_conv = np.concatenate(np.zeros([input_tensor.shape[2], 1]))
                    for k in range(input_tensor.shape[1]):
                        output_single_channel = correlate(input_tensor[i, k], self.weights[j, k], 'same')
                        output_single_conv += output_single_channel
                    output_single_batch.append(output_single_conv)
                output_tensor.append(output_single_batch)
            output_tensor = np.array(output_tensor)

            stride_shape = self.stride_shape
            strided_output_tensor = np.zeros([output_tensor.shape[0], output_tensor.shape[1], 
                                              ((output_tensor.shape[2] - 1) // stride_shape[0]) + 1])

            for i in range(output_tensor.shape[0]):
                for j in range(output_tensor.shape[1]):
                    y = 0
                    for k in range(0, output_tensor.shape[2], stride_shape[0]):
                        strided_output_tensor[i][j][y] = output_tensor[i][j][k]
                        y += 1

        return strided_output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass of the convolutional layer.

        Computes gradients for the weights and propagates the error backward through the layer.

        Args:
            error_tensor (numpy.ndarray): Gradient of the loss function with respect to the output.

        Returns:
            numpy.ndarray: The propagated error tensor for previous layers.
        """
        weights_backward = []
        for i in range(self.convolution_shape[0]):
            weights_backward_single_kernel = []
            for j in range(self.num_kernels):
                weights_backward_single_kernel.append(self.weights[j][i])
            weights_backward.append(weights_backward_single_kernel)
        weights_backward = np.array(weights_backward)
        self.weights_backward = weights_backward

        if len(error_tensor.shape) == 4:
            stride_shape = self.stride_shape if len(self.stride_shape) == 2 else np.concatenate(self.stride_shape, self.stride_shape)
            unstrided_error_tensor = np.zeros([self.input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]])

            for i in range(unstrided_error_tensor.shape[0]):
                for j in range(unstrided_error_tensor.shape[1]):
                    for k in range(error_tensor.shape[2]):
                        for l in range(error_tensor.shape[3]):
                            unstrided_error_tensor[i][j][k * stride_shape[0]][l * stride_shape[1]] = error_tensor[i][j][k][l]

        self.gradient_bias = np.zeros([error_tensor.shape[1]])
        for i in range(error_tensor.shape[0]):
            for j in range(error_tensor.shape[1]):
                self.gradient_bias[j] += np.sum(error_tensor[i][j])

        if hasattr(self, 'optimizer'):
            if not hasattr(self, 'weights_optimizer'):
                self.weights_optimizer = copy.deepcopy(self.optimizer)
                self.bias_optimizer = copy.deepcopy(self.optimizer)
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return unstrided_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initializes the weights and biases using the specified initializers.

        Args:
            weights_initializer: A weight initialization method.
            bias_initializer: A bias initialization method.
        """
        self.weights = weights_initializer.initialize(
            (self.num_kernels,) + self.convolution_shape, 
            self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2],
            self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]
        )
        self.bias = bias_initializer.initialize(self.num_kernels, self.convolution_shape[0], 1)
