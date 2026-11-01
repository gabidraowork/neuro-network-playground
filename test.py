import numpy as np
import pickle
from scipy.io import loadmat
from dense import Dense
from tanh import Tanh
from ReLU import Relu
from softmax import Softmax

mnist = loadmat(r"input/mnist-original.mat")
mnist_data = np.array(mnist["data"].T)
mnist_label = np.array(mnist["label"]).reshape(-1)

network = [
    Dense(mnist_data.shape[1], 16),
    Relu(),
    Dense(16, 16),
    Relu(),
    Dense(16, 10),
    Softmax()
]

with open("mnist_model.pkl", "rb") as file:
    model_data = pickle.load(file)

dense_layers = [
    layer for layer in network
    if isinstance(layer, Dense)
]

for layer, saved_layer in zip(dense_layers, model_data):

    layer.weights = saved_layer["weights"]
    layer.bias = saved_layer["bias"]

test_sample = 66666

d = mnist_data[test_sample]

output = d.reshape(784, 1)

for layer in network:
    output = layer.foward(output)

print(output)
predicao = np.argmax(output)

print("Previsão:", predicao)
print("Correto:", mnist_label[test_sample])