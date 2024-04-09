import numpy as np


def sigmoid_normalization(x, threshold = 0.0, std = 1.0):
    return 1 / (1 + np.exp(- (x - threshold) / std))