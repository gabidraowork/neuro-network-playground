import numpy as np

def one_hot(label):
    excepted = np.zeros((10,1))
    excepted[int(label)] = 1
    return excepted