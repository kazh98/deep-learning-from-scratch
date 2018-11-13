#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from struct import unpack_from
from gzip import GzipFile
from typing import Tuple


def _load_image_file(path: str) -> np.ndarray:
    with GzipFile(path, 'rb') as fp:
        assert fp.read(4) == b'\x00\x00\x08\x03', 'Invalid file format'
        n, rows, columns = unpack_from('>III', fp.read(12))
        r = np.array(list(fp.read(n * rows * columns)))
        r = r.reshape((n, rows * columns))
    return r


def _load_label_file(path: str, one_hot_label: bool=False) -> np.ndarray:
    with GzipFile(path, 'rb') as fp:
        assert fp.read(4) == b'\x00\x00\x08\x01', 'Invalid file format'
        n = unpack_from('>I', fp.read(4))[0]
        r = np.array(list(fp.read(n)), dtype=np.byte)
    if one_hot_label:
        indices = r
        r = np.zeros((indices.shape[0], 10), dtype=np.byte)
        for i, j in enumerate(indices):
            r[i, j] = 1
    return r


def load_train_data(one_hot_label: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    return _load_image_file('./data/train-images-idx3-ubyte.gz'), \
           _load_label_file('./data/train-labels-idx1-ubyte.gz', one_hot_label)


def load_test_data(one_hot_label: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    return _load_image_file('./data/t10k-images-idx3-ubyte.gz'), \
           _load_label_file('./data/t10k-labels-idx1-ubyte.gz', one_hot_label)
