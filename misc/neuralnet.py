#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from .functions import sigmoid, softmax, cross_entropy_error
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        u = np.dot(x, W1) + b1
        u = sigmoid(u)
        u = np.dot(u, W2) + b2
        u = softmax(u)
        return u

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        return cross_entropy_error(self.predict(x), t)

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
