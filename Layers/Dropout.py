from Layers.Base import BaseLayer
import numpy as np
class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase == False:
            random_numbers = np.random.uniform(0, 1, input_tensor.shape)
            dropout_matrix = random_numbers < self.probability
            output_tensor = np.multiply(dropout_matrix, input_tensor)
            output_tensor = np.multiply(output_tensor, 1/self.probability)
            self.dropout_matrix = dropout_matrix
            return output_tensor

        if self.testing_phase == True:
            output_tensor = input_tensor
            return output_tensor



    def backward(self, error_tensor):
        if self.testing_phase == False:
            output_error = np.multiply(self.dropout_matrix, error_tensor)
            output_error = np.multiply(output_error, 1 / self.probability)
            return output_error
