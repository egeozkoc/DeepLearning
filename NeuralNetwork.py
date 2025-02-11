import Layers.FullyConnected
from Optimization.Loss import CrossEntropyLoss
import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = copy.deepcopy(optimizer)
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = CrossEntropyLoss()
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def next(self):
        pass

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()

        regularization_loss = 0
        for i in range(0, len(self.layers)):
            self.layer = self.layers[i]
            input_tensor = self.layer.forward(input_tensor)
            if isinstance(self.layer, Layers.FullyConnected.FullyConnected) or isinstance(self.layer, Layers.RNN.RNN):
                if hasattr(self.layer, 'optimizer'):
                    if hasattr(self.layer.optimizer, 'regularizer'):
                        regularization_loss += self.layer.calculate_regularization_loss()

        loss = self.loss_layer.forward(input_tensor, self.label_tensor) + regularization_loss

        return loss

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)

        for i in range(len(self.layers) - 1, -1, -1):
            self.layer = self.layers[i]
            error = self.layer.backward(error)

        return error


    def append_layer(self, layer):
        if layer.trainable is True:

            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase(False)
        for i in range(0, iterations):
            self.forward()
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase(True)
        for i in range(0, len(self.layers)):
            self.layer = self.layers[i]
            input_tensor = self.layer.forward(input_tensor)
        return input_tensor

    def phase(self, testing_phase):
        for i in range(0, len(self.layers)):
            self.layers[i].testing_phase = testing_phase









