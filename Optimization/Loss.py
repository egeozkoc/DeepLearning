import numpy as np

class CrossEntropyLoss:
    """
    Implements the Cross-Entropy Loss function for classification tasks.
    """
    
    def __init__(self):
        """
        Initializes the CrossEntropyLoss class, setting a small epsilon value to avoid numerical instability.
        """
        self.eps = np.finfo(np.float64).eps

    def forward(self, prediction_tensor, label_tensor):
        """
        Computes the forward pass of the Cross-Entropy Loss.

        Args:
            prediction_tensor (numpy.ndarray): Predicted probabilities.
            label_tensor (numpy.ndarray): Ground truth labels (one-hot encoded).

        Returns:
            float: Computed cross-entropy loss.
        """
        loss = np.sum((-np.multiply(np.log(prediction_tensor + self.eps), label_tensor)))
        self.prediction_tensor = prediction_tensor
        return loss

    def backward(self, label_tensor):
        """
        Computes the gradient of the loss with respect to the predictions.

        Args:
            label_tensor (numpy.ndarray): Ground truth labels (one-hot encoded).

        Returns:
            numpy.ndarray: Error tensor representing the gradient.
        """
        error_tensor = -np.divide(label_tensor, (self.prediction_tensor + self.eps))
        return error_tensor