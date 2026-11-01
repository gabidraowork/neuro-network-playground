#Solving XOR
from dense import Dense
from tanh import Tanh
from mse import mse, mse_prime
import numpy as np

X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4, 2, 1))
Y = np.reshape([ [0], [1], [1], [0] ], (4, 1, 1 ))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 1000
learning_rate = 0.1

#treino

for e in range(epochs):
    error = 0
    for x,y in zip(X,Y):
        # foward
        output = x

        for layer in network:
            output = layer.foward(output)
        
        error += mse(y, output)

        #backward

        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

        
        formatted_output = np.array2string(
        output,
        formatter={'float_kind': lambda n: f'{n:.8f}'}
        )
        formatted_error = np.array2string(
            error,
            formatter={'float_kind': lambda n: f'{n:.15f}'}
        )

        error /= len(X)
        print(f'input={x[0]}|{x[1]}, output={formatted_output}, expected={y}, error={formatted_error}')
input = np.reshape([[float(input("Primeiro digito: ")), float(input("Segundo digito: "))]],(2,1))

output = input
for layer in network:
    output = layer.foward(output)

formatted_output = np.array2string(
    output,
    formatter={'float_kind': lambda n: f'{n:.8f}'}
    )

print(f'input={input[0]}|{input[1]}, output={formatted_output}')
