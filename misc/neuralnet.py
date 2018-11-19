#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
from .layers import *
from typing import Callable

__eps = 1e-4


def _numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray):
    g = np.zeros_like(x)
    for i in range(x.size):
        t = x.flat[i]
        x.flat[i] = t + __eps
        fxh1 = f(x)
        x.flat[i] = t - __eps
        fxh2 = f(x)
        x.flat[i] = t
        g.flat[i] = (fxh1 - fxh2) / (2 * __eps)
    return g


class SimpleNet(object):
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.W)

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        u = self.predict(x)
        u = softmax(u)
        return cross_entropy_error(u, t)

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return _numerical_gradient(lambda w: self.loss(x, t), self.W)


class TwoLayerNet(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        x = self.predict(x)
        return self.lastLayer.forward(x, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def score(self, x: np.ndarray, t: np.ndarray) -> float:
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / x.shape[0]

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return {
            'W1': _numerical_gradient(lambda w: self.loss(x, t), self.params['W1']),
            'b1': _numerical_gradient(lambda w: self.loss(x, t), self.params['b1']),
            'W2': _numerical_gradient(lambda w: self.loss(x, t), self.params['W2']),
            'b2': _numerical_gradient(lambda w: self.loss(x, t), self.params['b2'])
        }

    def gradient(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.loss(x, t)
        dout = self.lastLayer.backward(1)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)
        return {
            'W1': self.layers['Affine1'].dW,
            'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW,
            'b2': self.layers['Affine2'].db
        }
