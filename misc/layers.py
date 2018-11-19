#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from .functions import softmax, cross_entropy_error


class Relu(object):
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        return dout


class Sigmoid(object):
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1. / (1. + np.exp(-x))
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.out * (1. - self.out)


class Affine(object):
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)


class SoftmaxWithLoss(object):
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1) -> np.ndarray:
        return (self.y - self.t) / self.t.shape[0]
