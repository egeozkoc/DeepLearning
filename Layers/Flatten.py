import numpy as np
from Layers.Base import BaseLayer
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        flattened_tensor = []
        self.original_shape = input_tensor.shape
        for i in range(0, input_tensor.shape[0]):
            flattened_tensor.append(input_tensor[i].flatten())
        input_tensor = np.array(flattened_tensor)
        return input_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.original_shape)
        return error_tensor


