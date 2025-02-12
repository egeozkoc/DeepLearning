class BaseLayer:
    """
    In this project we implement a layer oriented Deep Learning framework for high level of abstraction.
    All the layers in the framework will inherit from the BaseLayer class.
    """
    
    def __init__(self):
        """
        Initializes the BaseLayer with default attributes.
        """
        self.trainable = False
        self.weights = None
        self.testing_phase = False
