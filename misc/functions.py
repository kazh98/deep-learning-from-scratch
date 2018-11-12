#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def step(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.fmax(0, x)


def identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


def softmax(x: np.ndarray) -> np.ndarray:
    u = x - np.max(x)
    np.exp(u, out=u)
    u /= np.sum(u)
    return u
