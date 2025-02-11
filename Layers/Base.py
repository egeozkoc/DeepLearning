class BaseLayer:
    """
    A base class for all layers in the neural network.
    """
    
    def __init__(self):
        """
        Initializes the BaseLayer with default attributes.
        """
        self.trainable = False
        self.weights = None
        self.testing_phase = False
