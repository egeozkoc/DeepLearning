
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.eps = np.finfo(np.float64).eps

    def forward(self, prediction_tensor, label_tensor):
        loss = np.sum((-np.multiply(np.log(prediction_tensor + self.eps), label_tensor)))
        self.prediction_tensor = prediction_tensor
        return loss

    def backward(self, label_tensor):
        error_tensor = -np.divide(label_tensor, (self.prediction_tensor + self.eps))
        return error_tensor


