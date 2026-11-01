from activation import Activation
import numpy as np

class Relu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)