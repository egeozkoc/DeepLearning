import numpy as np

class Optimizer:
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if hasattr(self, 'regularizer'):
            weight_tensor = weight_tensor - self.regularizer.calculate_gradient(weight_tensor) * self.learning_rate
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0
    def calculate_update(self, weight_tensor, gradient_tensor):
        if hasattr(self, 'regularizer'):
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        updated_weight = self.v + weight_tensor
        return updated_weight

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self. r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if hasattr(self, 'regularizer'):
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + np.multiply(((1 - self.rho) * gradient_tensor), gradient_tensor)
        self.v_corrected = self.v / (1 - self.mu ** self.k)
        self.r_corrected = self.r / (1 - self.rho ** self.k)
        updated_weight = weight_tensor - self.learning_rate * self.v_corrected / (np.sqrt(self.r_corrected) + 1e-8)
        self.k += 1
        return updated_weight