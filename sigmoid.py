from activation import Activation
import numpy as np

class Sigmoid(Activation):
    def __init__(self):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_prime = lambda x: self.sigmoid(x)*(1-self.sigmoid(x))
        super().__init__(self.sigmoid, self.sigmoid_prime) 