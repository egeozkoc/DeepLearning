from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Optimization.Optimizers import Sgd
import numpy as np
class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable = True
        self.memorize = False
        self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc2 = FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.weights = self.fc1.weights
        self.optimizer = None

    def forward(self, input_tensor):
        self.weights = self.fc1.weights
        if self.memorize == False:
            self.hidden_state = np.zeros(self.hidden_size)
        self.fc1_input = np.zeros([input_tensor.shape[0], self.input_size + self.hidden_size + 1])
        self.fc2_input = np.zeros([input_tensor.shape[0], self.hidden_size + 1])
        self.tanh_output = np.zeros([input_tensor.shape[0], self.hidden_size])
        self.sigmoid_output = np.zeros([input_tensor.shape[0], self.output_size])
        output_tensor = np.zeros([input_tensor.shape[0], self.output_size])
        for i in range(0, input_tensor.shape[0]):
            input_tensor_tilda = np.concatenate((input_tensor[i], self.hidden_state))
            input_tensor_tilda = input_tensor_tilda.reshape(1, -1)
            self.hidden_state = self.tanh.forward(self.fc1.forward(input_tensor_tilda))
            output_tensor[i] = self.sigmoid.forward(self.fc2.forward(self.hidden_state))
            self.hidden_state = self.hidden_state.ravel()
            self.fc1_input[i] = self.fc1.get_input()
            self.fc2_input[i] = self.fc2.get_input()
            self.tanh_output[i] = self.tanh.get_output()
            self.sigmoid_output[i] = self.sigmoid.get_output()
        return output_tensor

    def backward(self, error_tensor):
        error_output = np.zeros((error_tensor.shape[0], self.input_size))
        self.gradient_hidden_state = 0
        self.gradient_weights = 0
        for i in range(error_tensor.shape[0]-1, -1, -1):
            self.fc1.set_input(self.fc1_input[i].reshape(1, -1))
            self.fc2.set_input(self.fc2_input[i].reshape(1, -1))
            self.tanh.set_output(self.tanh_output[i].reshape(1, -1))
            self.sigmoid.set_output(self.sigmoid_output[i].reshape(1, -1))
            backward_step = self.sigmoid.backward(error_tensor[i])
            backward_step = self.fc2.backward(backward_step)
            backward_step += self.gradient_hidden_state
            backward_step = self.tanh.backward(backward_step)
            backward_step = self.fc1.backward(backward_step)
            error_output[i] = backward_step[:, 0:self.input_size]
            self.gradient_hidden_state = backward_step[:, self.input_size::]
            self.gradient_weights += self.fc1.gradient_weights
        if self.optimizer is not None:
            self.set_weights(self.optimizer.calculate_update(self.get_weights(), self.gradient_weights))
        return error_output

    def initialize(self, weights_initializer, bias_initializer):
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)
        self.weights = self.fc1.weights

    def get_weights(self):
        return self.fc1.weights

    def set_weights(self, weights):
        self.fc1.weights = weights
        self.weights = self.fc1.weights

    @property
    def weights(self):
        return self.fc1.weights

    @weights.setter
    def weights(self, weights):
        self.fc1.weights = weights

    def calculate_regularization_loss(self):
        regularization_loss = self.optimizer.regularizer.norm(self.weights)
        return regularization_loss




