#!/usr/bin/env python3
import numpy as np

if __name__ == '__main__':
    data = np.loadtxt('data/train.csv', delimiter=',',
                      skiprows=1, dtype=np.uint8)
    np.save('data/train_X', data[:, 1:])
    np.save('data/train_y', data[:, 0])
    data = np.loadtxt('data/test.csv', delimiter=',',
                      skiprows=1, dtype=np.uint8)
    np.save('data/test_X', data)
