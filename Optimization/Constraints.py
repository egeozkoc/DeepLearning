import numpy as np

class L2_Regularizer:
    """
    Implements L2 regularization, also known as weight decay.
    """
    
    def __init__(self, alpha):
        """
        Initializes the L2 Regularizer.

        Args:
            alpha (float): Regularization strength.
        """
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        Computes the gradient of the L2 regularization term.

        Args:
            weights (numpy.ndarray): Weight matrix.

        Returns:
            numpy.ndarray: Gradient of the regularization term.
        """
        return weights * self.alpha
    
    def norm(self, weights):
        """
        Computes the L2 regularization loss.

        Args:
            weights (numpy.ndarray): Weight matrix.

        Returns:
            float: L2 regularization loss.
        """
        return np.sum(np.square(weights)) * self.alpha

class L1_Regularizer:
    """
    Implements L1 regularization, which encourages sparsity in weights.
    """
    
    def __init__(self, alpha):
        """
        Initializes the L1 Regularizer.

        Args:
            alpha (float): Regularization strength.
        """
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        """
        Computes the gradient of the L1 regularization term.

        Args:
            weights (numpy.ndarray): Weight matrix.

        Returns:
            numpy.ndarray: Gradient of the regularization term.
        """
        return np.sign(weights) * self.alpha
    
    def norm(self, weights):
        """
        Computes the L1 regularization loss.

        Args:
            weights (numpy.ndarray): Weight matrix.

        Returns:
            float: L1 regularization loss.
        """
        return np.sum(np.abs(weights)) * self.alpha
