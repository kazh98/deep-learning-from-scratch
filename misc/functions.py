#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union

__eps = 1e-7


def step(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.fmax(0, x)


def identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        u = x - np.max(x, axis=1)[:, np.newaxis]
        np.exp(u, out=u)
        u /= np.sum(u, axis=1)[:, np.newaxis]
        return u
    u = x - np.max(x)
    np.exp(u, out=u)
    u /= np.sum(u)
    return u


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> Union[float, np.ndarray]:
    if y.ndim == 1:
        u = y - t
        return np.inner(u, u) / 2
    u = y - t
    return np.einsum('ij,ij->', u, u) / 2 / y.shape[0]


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> Union[float, np.ndarray]:
    if y.ndim == 1:
        return -np.inner(t, np.log(y + __eps))
    return -np.sum(t * np.log(y + __eps)) / y.shape[0]
