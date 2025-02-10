from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        output_tensor = np.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] - self.pooling_shape[0] + 1, input_tensor.shape[3] - self.pooling_shape[1] + 1])
        max_position = np.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] - self.pooling_shape[0] + 1, input_tensor.shape[3] - self.pooling_shape[1] + 1])
        for i in range(0, input_tensor.shape[0]):
            for j in range(0, input_tensor.shape[1]):
                for k in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1):
                    for l in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1):
                        submatrix = input_tensor[i][j][k : k + self.pooling_shape[0], l : l + self.pooling_shape[1]]
                        output_tensor[i][j][k][l] = np.max(submatrix)
                        max_position[i][j][k][l] = np.argmax(submatrix)

        strided_output_tensor = np.zeros(
            [output_tensor.shape[0], output_tensor.shape[1], ((output_tensor.shape[2] - 1) // self.stride_shape[0]) + 1,
             ((output_tensor.shape[3] - 1) // self.stride_shape[1]) + 1])

        strided_max_position = np.zeros(
            [output_tensor.shape[0], output_tensor.shape[1], ((output_tensor.shape[2] - 1) // self.stride_shape[0]) + 1,
             ((output_tensor.shape[3] - 1) // self.stride_shape[1]) + 1])

        for i in range(0, output_tensor.shape[0]):
            for j in range(0, output_tensor.shape[1]):
                y = 0
                for k in range(0, output_tensor.shape[2], self.stride_shape[0]):
                    x = 0
                    for l in range(0, output_tensor.shape[3], self.stride_shape[1]):
                        strided_output_tensor[i][j][y][x] = output_tensor[i][j][k][l]
                        strided_max_position[i][j][y][x] = max_position[i][j][k][l]
                        x += 1
                    y += 1

        self.max_position = strided_max_position
        return strided_output_tensor


    def backward(self, error_tensor):
        previous_error_tensor = np.zeros(self.input_shape)
        for i in range(0, error_tensor.shape[0]):
            for j in range(0, error_tensor.shape[1]):
                for k in range(0, error_tensor.shape[2]):
                    for l in range(0, error_tensor.shape[3]):
                        x = int(k * self.stride_shape[0] + self.max_position[i][j][k][l] // self.pooling_shape[1])
                        y = int(l * self.stride_shape[1] + self.max_position[i][j][k][l] % self.pooling_shape[1])
                        previous_error_tensor[i][j][x][y] += error_tensor[i][j][k][l]
        return previous_error_tensor







