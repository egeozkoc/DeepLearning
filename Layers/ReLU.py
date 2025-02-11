from Layers.Base import BaseLayer
import numpy as np

def mapReLU(input):

    if input > 0:
        output = input
    else:
        output = 0

    return output

class ReLU(BaseLayer):
    def __int__(self):
        super().__init__()
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        vReLU = np.vectorize(mapReLU)
        output_tensor = vReLU(input_tensor)
        return output_tensor


    def backward(self, error_tensor):
        vReLU = np.vectorize(mapReLU)
        output_error_tensor = vReLU(self.input_tensor)
        output_error_tensor = output_error_tensor > 0
        output_error_tensor = np.multiply(output_error_tensor, error_tensor)
        return output_error_tensor





