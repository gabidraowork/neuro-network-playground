import numpy as np
from scipy.io import loadmat
from dense import Dense
from softmax import Softmax, func_Softamx
from tanh import Tanh
from sigmoid import Sigmoid
from ReLU import Relu
from mse import mse, mse_prime
from one_hot import one_hot
import pickle
import random

mnist = loadmat(r"input/mnist-original.mat")
mnist_data = np.array(mnist["data"].T)
mnist_label = np.array(mnist["label"]).reshape(-1)
mnist_data = mnist_data.astype(np.float64) / 255.0

indices = np.random.permutation(70000)

#Embaralha os dados para evitar catastrophic forgetting
mnist_data = mnist_data[indices]
mnist_label = mnist_label[indices]

print(mnist_data.shape[1])

network = [
    Dense(mnist_data.shape[1], 16),
    Relu(),
    Dense(16, 16),
    Relu(),
    Dense(16, 10),
    Softmax()
]

epochs = 3
learning_rate = 0.05

for e in range(epochs):
    error = 0
    for index, (d, l) in enumerate(zip(mnist_data, mnist_label)):
        #foward
        output = np.reshape(d, (784,1))
        for layer in network:
            output = layer.foward(output)

        error = mse(one_hot(l), func_Softamx(output))

        #backward

        grad = mse_prime(one_hot(l), func_Softamx(output))
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

        if index % 10000 == 0:
            print(f"Accuracy: {formatted_error}, guess: {np.argmax(output)}, expected:{l}")
       
    # error /= len(mnist_data)

print("FINAL TEST")
for _ in range(100):        
    test_sample = random.randint(1,70000)
    d = mnist_data[test_sample]
    output = np.reshape(d, (784,1))
    for layer in network:
        output = layer.foward(output)
    error = mse(one_hot(mnist_label[test_sample]), func_Softamx(output))
    output = func_Softamx(output)

    print(f"SAMPLE: {test_sample}, EXPECTED: {mnist_label[test_sample]}, OUTPUT: {np.argmax(output)}, error: {error}")

model_data = [
    {"weights": l.weights, "bias": l.bias}
    for l in network if isinstance(l, Dense)
]
with open("mnist_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("Modelo Salvo")
  


    
