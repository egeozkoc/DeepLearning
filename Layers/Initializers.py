import numpy as np

class Constant:
    """
    Implements a constant weight initializer, filling weights with a fixed value.
    """
    
    def __init__(self, value):
        """
        Initializes the Constant initializer.

        Args:
            value (float): The constant value to initialize weights.
        """
        self.value = value
    
    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Generates a weight tensor filled with the constant value.

        Args:
            weights_shape (tuple): Shape of the weight tensor.
            fan_in (int): Number of input connections.
            fan_out (int): Number of output connections.

        Returns:
            numpy.ndarray: Initialized weight tensor.
        """
        return np.full(weights_shape, self.value)

class UniformRandom:
    """
    Implements a uniform random weight initializer.
    """
    
    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Generates a weight tensor with values sampled from a uniform distribution.

        Args:
            weights_shape (tuple): Shape of the weight tensor.
            fan_in (int): Number of input connections.
            fan_out (int): Number of output connections.

        Returns:
            numpy.ndarray: Initialized weight tensor.
        """
        return np.random.uniform(low=0.0, high=1.0, size=weights_shape)

class Xavier:
    """
    Implements Xavier (Glorot) weight initialization, suitable for tanh activations.
    """
    
    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Generates a weight tensor using Xavier initialization.

        Args:
            weights_shape (tuple): Shape of the weight tensor.
            fan_in (int): Number of input connections.
            fan_out (int): Number of output connections.

        Returns:
            numpy.ndarray: Initialized weight tensor.
        """
        std_dev = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(loc=0.0, scale=std_dev, size=weights_shape)

class He:
    """
    Implements He weight initialization, suitable for ReLU activations.
    """
    
    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Generates a weight tensor using He initialization.

        Args:
            weights_shape (tuple): Shape of the weight tensor.
            fan_in (int): Number of input connections.
            fan_out (int): Number of output connections.

        Returns:
            numpy.ndarray: Initialized weight tensor.
        """
        std_dev = np.sqrt(2 / fan_in)
        return np.random.normal(loc=0.0, scale=std_dev, size=weights_shape)
