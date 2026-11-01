import numpy as np


class Softmax:

    def foward(self, input):
        self.input = input

        # estabilidade numérica
        exp_values = np.exp(input - np.max(input))

        self.output = exp_values / np.sum(exp_values)

        return self.output


    def backward(self, output_gradient, learning_rate):

        jacobian = (
            np.diagflat(self.output)
            - self.output @ self.output.T
        )

        return jacobian @ output_gradient

def func_Softamx(input):
    # estabilidade numérica
    exp_values = np.exp(input - np.max(input))
    
    output = exp_values / np.sum(exp_values)
    
    return output