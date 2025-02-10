import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.eps = 1e-16
        self.decay = 0.8
        self.test_mean = 0
        self.test_var = 0
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self.testing_phase = False


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.channels,), self.channels, self.channels)
        self.bias = bias_initializer.initialize((self.channels,), self.channels, self.channels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape
        if self.testing_phase == False:
            if len(self.input_shape) == 4:
                input_tensor = self.reformat(input_tensor)
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            self.input_tensor_tilda = (input_tensor - self.mean)/np.sqrt(self.var + self.eps)
            output_tensor = np.multiply(self.weights, self.input_tensor_tilda) + self.bias
            if len(self.input_shape) == 4:
                output_tensor = self.reformat(output_tensor)
            self.test_mean = self.decay * self.mean + (1 - self.decay) * self.mean
            self.test_var = self.decay * self.var + (1 - self.decay) * self.var

        if self.testing_phase == True:
            if len(self.input_shape) == 4:
                input_tensor = self.reformat(input_tensor)
            input_tensor_tilda = (input_tensor - self.test_mean) / np.sqrt(self.test_var + self.eps)
            output_tensor = np.multiply(self.weights, input_tensor_tilda) + self.bias
            if len(self.input_shape) == 4:
                output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):
        self.error_shape = error_tensor.shape
        input_tensor = self.input_tensor
        if len(self.error_shape) == 4:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(input_tensor)
        gradient_input = compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean, self.var)
        if len(self.error_shape) == 4:
            gradient_input = self.reformat(gradient_input)
        self.gradient_weights = 0
        self.gradient_bias = 0
        for i in range(0, error_tensor.shape[0]):
            self.gradient_weights += np.multiply(error_tensor[i], self.input_tensor_tilda[i])
            self.gradient_bias += error_tensor[i]
        if hasattr(self, 'optimizer'):
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return gradient_input

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            output = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]))
            output = np.transpose(output, (0, 2, 1))
            output = np.reshape(output, (output.shape[0] * output.shape[1], output.shape[2]))

        if len(tensor.shape) == 2:
            output = np.reshape(tensor, (self.input_shape[0], self.input_shape[2] * self.input_shape[3], self.input_shape[1]))
            output = np.transpose(output, (0, 2, 1))
            output = np.reshape(output, (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        return output



