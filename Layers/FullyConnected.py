from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.optimizer = None
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

    def forward(self, input_tensor):
        bias = np.ones([input_tensor.shape[0], 1])
        self.input_tensor = np.concatenate([input_tensor, bias], axis = 1)
        output_tensor = np.matmul(self.input_tensor, self.weights)
        self.batch_size = output_tensor.shape[0]
        self.output_size = output_tensor.shape[1]
        return output_tensor

    def backward(self, error_tensor):
        previous_error_tensor = np.matmul(error_tensor, np.transpose(self.weights))
        previous_error_tensor = np.delete(previous_error_tensor, -1, 1)
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return previous_error_tensor

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_optimizer(self):
        return self.optimizer

    def set_input(self, input_tensor):
        self.input_tensor = input_tensor

    def get_input(self):
        return self.input_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.concatenate([self.weights, self.bias], axis = 0)

    def calculate_regularization_loss(self):
        regularization_loss = self.optimizer.regularizer.norm(self.weights)
        return regularization_loss









