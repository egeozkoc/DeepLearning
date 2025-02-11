import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __int__(self):
        super().__init__()


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        max_x = np.max(input_tensor, axis=1)
        max_x = max_x.reshape(-1,1)
        input_tensor = input_tensor - max_x
        exp_input_tensor = np.exp(input_tensor)
        sum_batch = np.sum(exp_input_tensor, axis=1)
        sum_batch = sum_batch.reshape(-1, 1)
        sum_batch = np.tile(sum_batch, exp_input_tensor.shape[1])
        output_tensor = np.divide(exp_input_tensor, sum_batch)
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        previous_error_tensor = np.multiply(self.output_tensor, (error_tensor - np.sum(np.multiply(error_tensor, self.output_tensor), axis = 1).reshape(-1, 1)))
        return previous_error_tensor


'''
input = np.array([[1,2,3],[4,6,8]])
a=SoftMax()
output = a.forward(input)'''







