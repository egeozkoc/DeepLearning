import numpy as np
from scipy.signal import convolve, correlate
from Layers.Base import BaseLayer
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, (self.num_kernels,) + self.convolution_shape)
        self.bias = np.random.uniform(0, 1, self.num_kernels)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 4:
            output_tensor = []
            for i in range(0, input_tensor.shape[0]):
                output_single_batch = []
                for j in range(0, self.num_kernels):
                    output_single_conv = np.zeros([input_tensor.shape[2], input_tensor.shape[3]])
                    for k in range(0, input_tensor.shape[1]):
                        output_single_channel = correlate(input_tensor[i][k], self.weights[j][k], 'same')
                        output_single_conv += output_single_channel
                    output_single_batch.append(output_single_conv + self.bias[j])
                output_tensor.append(output_single_batch)
            output_tensor = np.array(output_tensor)

            if len(self.stride_shape) == 1:
                stride_shape = np.concatenate(self.stride_shape, self.stride_shape)
            elif len(self.stride_shape) == 2:
                stride_shape = self.stride_shape

            strided_output_tensor = np.zeros([output_tensor.shape[0], output_tensor.shape[1], ((output_tensor.shape[2] - 1) // stride_shape[0]) + 1, ((output_tensor.shape[3] - 1) // stride_shape[1]) + 1])

            for i in range(0, output_tensor.shape[0]):
                for j in range(0, output_tensor.shape[1]):
                    y = 0
                    for k in range(0, output_tensor.shape[2], stride_shape[0]):
                        x = 0
                        for l in range(0, output_tensor.shape[3], stride_shape[1]):
                            strided_output_tensor[i][j][y][x] = output_tensor[i][j][k][l]
                            x += 1
                        y += 1

        elif len(input_tensor.shape) == 3:
            output_tensor = []
            for i in range(0, input_tensor.shape[0]):
                output_single_batch = []
                for j in range(0, self.num_kernels):
                    output_single_conv = np.concatenate(np.zeros([input_tensor.shape[2], 1]))
                    for k in range(0, input_tensor.shape[1]):
                        output_single_channel = correlate(input_tensor[i, k], self.weights[j, k], 'same')
                        output_single_conv += output_single_channel
                    output_single_batch.append(output_single_conv)
                output_tensor.append(output_single_batch)
            output_tensor = np.array(output_tensor)

            stride_shape = self.stride_shape
            strided_output_tensor = np.zeros(
                [output_tensor.shape[0], output_tensor.shape[1], ((output_tensor.shape[2] - 1) // stride_shape[0]) + 1])

            for i in range(0, output_tensor.shape[0]):
                for j in range(0, output_tensor.shape[1]):
                    y = 0
                    for k in range(0, output_tensor.shape[2], stride_shape[0]):
                        strided_output_tensor[i][j][y] = output_tensor[i][j][k]
                        y += 1

        return strided_output_tensor

    def backward(self, error_tensor):
        weights_backward = []
        for i in range(0, self.convolution_shape[0]):
            weights_backward_single_kernel = []
            for j in range(0, self.num_kernels):
                weights_backward_single_kernel.append(self.weights[j][i])
            weights_backward.append(weights_backward_single_kernel)
        weights_backward = np.array(weights_backward)
        self.weights_backward = weights_backward



        if len(error_tensor.shape) == 4:

            if len(self.stride_shape) == 1:
                stride_shape = np.concatenate(self.stride_shape, self.stride_shape)
            elif len(self.stride_shape) == 2:
                stride_shape = self.stride_shape

            unstrided_error_tensor = np.zeros([self.input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]])
            for i in range(0, unstrided_error_tensor.shape[0]):
                for j in range(0, unstrided_error_tensor.shape[1]):
                    for k in range(0, error_tensor.shape[2]):
                        for l in range(0, error_tensor.shape[3]):
                            unstrided_error_tensor[i][j][k * stride_shape[0]][l * stride_shape[1]] = error_tensor[i][j][k][l]



            output_error_tensor = []
            for i in range(0, unstrided_error_tensor.shape[0]):
                output_error_single_batch = []
                for j in range(0, self.convolution_shape[0]):
                    output_error_single_conv = np.zeros([self.input_tensor.shape[2], self.input_tensor.shape[3]])
                    for k in range(0, unstrided_error_tensor.shape[1]):
                        if not isinstance(self.weights_backward[j][k], np.ndarray):
                            weight = np.array(self.weights_backward[j][k]).reshape([1,1])
                            output_error_single_channel = convolve(unstrided_error_tensor[i][k], weight, 'same')
                        else:
                            output_error_single_channel = convolve(unstrided_error_tensor[i][k], self.weights_backward[j][k], 'same')
                        output_error_single_conv += output_error_single_channel
                    output_error_single_batch.append(output_error_single_conv)
                output_error_tensor.append(output_error_single_batch)
            output_error_tensor = np.array(output_error_tensor)

            gradient_weights = []
            for i in range(0, self.num_kernels):
                gradient_weights_single_kernel = []
                for j in range(0, self.convolution_shape[0]):
                    gradient_weights_single_conv = 0
                    for k in range(0, self.input_tensor.shape[0]):
                        input_channel = np.pad(self.input_tensor[k][j],
                                            ((self.convolution_shape[1] // 2, self.convolution_shape[1] // 2), (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)))
                        gradient_weights_single_conv += correlate(input_channel, unstrided_error_tensor[k][i], 'valid')
                    gradient_weights_single_kernel.append(gradient_weights_single_conv)
                gradient_weights.append(gradient_weights_single_kernel)
            gradient_weights = np.array(gradient_weights)
            self.gradient_weights = gradient_weights[:, :, 0:self.convolution_shape[1], 0:self.convolution_shape[2]]




        elif len(error_tensor.shape) == 3:

            unstrided_error_tensor = np.zeros([self.input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2]])
            for i in range(0, unstrided_error_tensor.shape[0]):
                for j in range(0, unstrided_error_tensor.shape[1]):
                    for k in range(0, unstrided_error_tensor.shape[2]):
                            unstrided_error_tensor[i][j][k] = error_tensor[i][j][k // self.stride_shape[0]]


            output_error_tensor = []
            for i in range(0, unstrided_error_tensor.shape[0]):
                output_error_single_batch = []
                for j in range(0, self.convolution_shape[0]):
                    output_error_single_conv = np.concatenate(np.zeros([unstrided_error_tensor.shape[2], 1]))
                    for k in range(0, unstrided_error_tensor.shape[1]):
                        output_error_single_channel = convolve(unstrided_error_tensor[i][k], self.weights_backward[j][k], 'same')
                        output_error_single_conv += output_error_single_channel
                    output_error_single_batch.append(output_error_single_conv)
                output_error_tensor.append(output_error_single_batch)
            output_error_tensor = np.array(output_error_tensor)

            gradient_weights = []
            for i in range(0, self.num_kernels):
                gradient_weights_single_kernel = []
                for j in range(0, self.convolution_shape[0]):
                    gradient_weights_single_conv = 0
                    for k in range(0, self.input_tensor.shape[0]):
                        input_channel = np.pad(self.input_tensor[k][j],
                                            (self.convolution_shape[1] // 2))
                        gradient_weights_single_conv += correlate(input_channel, unstrided_error_tensor[k][i], 'valid')
                    gradient_weights_single_kernel.append(gradient_weights_single_conv)
                gradient_weights.append(gradient_weights_single_kernel)
            gradient_weights = np.array(gradient_weights)
            self.gradient_weights = gradient_weights[:, :, 0:self.convolution_shape[1]]


        self.gradient_bias = np.zeros([error_tensor.shape[1]])
        for i in range(0, error_tensor.shape[0]):
            for j in range(0, error_tensor.shape[1]):
                self.gradient_bias[j] += np.sum(error_tensor[i][j])

        if hasattr(self, 'optimizer'):
            if not hasattr(self, 'weights_optimizer'):
                self.weights_optimizer = copy.deepcopy(self.optimizer)
                self.bias_optimizer = copy.deepcopy(self.optimizer)
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)
        return output_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.num_kernels,) + self.convolution_shape, self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2],
                                                      self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2])
        self.bias = bias_initializer.initialize(self.num_kernels, self.convolution_shape[0], 1)


