#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from struct import unpack_from
from gzip import GzipFile
from typing import Tuple


def load_image_file(path: str) -> np.ndarray:
    with GzipFile(path, 'rb') as fp:
        assert fp.read(4) == b'\x00\x00\x08\x03', 'Invalid file format'
        n, rows, columns = unpack_from('>III', fp.read(12))
        r = np.array(list(fp.read(n * rows * columns)))
        r = r.reshape((n, rows * columns))
    return r


def load_label_file(path: str) -> np.ndarray:
    with GzipFile(path, 'rb') as fp:
        assert fp.read(4) == b'\x00\x00\x08\x01', 'Invalid file format'
        n = unpack_from('>I', fp.read(4))[0]
        r = np.array(list(fp.read(n)), dtype=np.byte)
    return r


def load_train_data() -> Tuple[np.ndarray, np.ndarray]:
    return load_image_file('./data/train-images-idx3-ubyte.gz'), load_label_file('./data/train-labels-idx1-ubyte.gz')


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    return load_image_file('./data/t10k-images-idx3-ubyte.gz'), load_label_file('./data/t10k-labels-idx1-ubyte.gz')
