import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms.
    """
    
    def add_regularizer(self, regularizer):
        """
        Adds a regularizer to the optimizer.

        Args:
            regularizer: Regularization function to be applied.
        """
        self.regularizer = regularizer

class Sgd(Optimizer):
    """
    Implements Stochastic Gradient Descent (SGD) optimization.
    """
    
    def __init__(self, learning_rate):
        """
        Initializes the SGD optimizer.

        Args:
            learning_rate (float): Learning rate for weight updates.
        """
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes the weight update using SGD.

        Args:
            weight_tensor (numpy.ndarray): Current weight values.
            gradient_tensor (numpy.ndarray): Gradient values.

        Returns:
            numpy.ndarray: Updated weight values.
        """
        if hasattr(self, 'regularizer'):
            weight_tensor -= self.regularizer.calculate_gradient(weight_tensor) * self.learning_rate
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum(Optimizer):
    """
    Implements SGD with momentum.
    """
    
    def __init__(self, learning_rate, momentum_rate):
        """
        Initializes the SGD with Momentum optimizer.

        Args:
            learning_rate (float): Learning rate for weight updates.
            momentum_rate (float): Momentum factor.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes the weight update using SGD with momentum.

        Args:
            weight_tensor (numpy.ndarray): Current weight values.
            gradient_tensor (numpy.ndarray): Gradient values.

        Returns:
            numpy.ndarray: Updated weight values.
        """
        if hasattr(self, 'regularizer'):
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return self.v + weight_tensor

class Adam(Optimizer):
    """
    Implements the Adam optimization algorithm.
    """
    
    def __init__(self, learning_rate, mu, rho):
        """
        Initializes the Adam optimizer.

        Args:
            learning_rate (float): Learning rate for weight updates.
            mu (float): Decay rate for first moment estimate.
            rho (float): Decay rate for second moment estimate.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Computes the weight update using Adam.

        Args:
            weight_tensor (numpy.ndarray): Current weight values.
            gradient_tensor (numpy.ndarray): Gradient values.

        Returns:
            numpy.ndarray: Updated weight values.
        """
        if hasattr(self, 'regularizer'):
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor
        v_corrected = self.v / (1 - self.mu ** self.k)
        r_corrected = self.r / (1 - self.rho ** self.k)
        updated_weight = weight_tensor - self.learning_rate * v_corrected / (np.sqrt(r_corrected) + 1e-8)
        self.k += 1
        return updated_weight