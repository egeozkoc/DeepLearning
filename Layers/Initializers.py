import numpy as np

class Constant:
    def __init__(self, value):
        self.value = value
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.full(weights_shape, self.value)
        return weights

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.uniform(low=0.0, high=1.0, size = weights_shape)
        return weights

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        std_dev = np.sqrt(2/(fan_out + fan_in))
        weights = np.random.normal(loc=0.0, scale=std_dev, size = weights_shape)
        return weights

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        std_dev = np.sqrt(2/fan_in)
        weights = np.random.normal(loc=0.0, scale=std_dev, size = weights_shape)
        return weights