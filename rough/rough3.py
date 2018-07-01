import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

_EPOCHS = 10
_TOTAL_ARRAYS = 1
_BATCH_SIZE = 1
_TIME1 = 5
_TIME2 = 5
_ENC_N = 30
_DEC_N = 30
_DEPTH = 10


if __name__ == '__main__':
	inp = (np.random.rand(_TOTAL_ARRAYS, _TIME1, 1) * 10).astype(int)
	oup = np.sort(inp, axis=1)
	ze = np.zeros((_TOTAL_ARRAYS, _TIME1, _DEPTH))
	for i in range(_BATCH_SIZE):
		for j in range(_TIME1):
			ze[i, j, inp[i, j]] = 1
	inp = ze
	ze = np.zeros((_TOTAL_ARRAYS, _TIME1, _DEPTH))
	for i in range(_BATCH_SIZE):
		for j in range(_TIME1):
			ze[i, j, oup[i, j]] = 1
	oup = ze
	print(inp)
	print(oup)
	print(ze)
	print(op)
