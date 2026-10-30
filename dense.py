from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def foward(self, input):
       self.input = input
       return self.weights @ self.input + self.bias

    def backward(self, output_gradient, learning_rate):
        #Calcula o gradiente dos parametros
        self.weights_gradient = output_gradient @ self.input.T
        #atualiza os parametros e os viéses por meio de Gradient descent, com step size fixo.
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * output_gradient
        #retorna a derivada do erro em relação ao input, para ser usado em outra layer
        return self.weights.T @ output_gradient