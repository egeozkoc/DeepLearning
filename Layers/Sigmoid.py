from Layers.Base import BaseLayer
import numpy as np
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        output_tensor = 1 / (1 + np.exp(-input_tensor))
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        derivative = np.multiply(self.output_tensor, (1 - self.output_tensor))
        output_error = error_tensor * derivative
        return output_error

    def get_output(self):
        return self.output_tensor

    def set_output(self, output_tensor):
        self.output_tensor = output_tensor
