import numpy as np
class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        gradient = weights * self.alpha
        return gradient
    def norm(self, weights):
        norm = np.sum(np.square(weights)) * self.alpha
        return norm

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate_gradient(self, weights):
        gradient = np.sign(weights) * self.alpha
        return gradient
    def norm(self, weights):
        norm = np.sum(np.abs(weights)) * self.alpha
        return norm
