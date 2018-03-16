#!/usr/bin/env python3
import numpy as np

if __name__ == '__main__':
    # load train
    data = np.loadtxt('data/train.csv', delimiter=',',
                      skiprows=1, dtype=np.uint8)
    x = data[:, 1:]
    x = x.reshape(x.shape[0], 28, 28)
    np.savez_compressed('data/train', x=x, y=data[:, 0])
    # load test
    data = np.loadtxt('data/test.csv', delimiter=',',
                      skiprows=1, dtype=np.uint8)
    x = data.reshape(data.shape[0], 28, 28)
    np.savez_compressed('data/test', x=x)
