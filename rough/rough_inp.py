import numpy as np

_EPOCHS = 10
_TOTAL_ARRAYS = 2
_BATCH_SIZE = 10
_TIME1 = 5
_TIME2 = 5
_ENC_N = 30
_DEC_N = 30
_DEPTH = 1

inp = np.random.rand(_TOTAL_ARRAYS, _TIME1, 1)
oup = np.sort(inp, axis=1)
z = np.zeros((oup.shape[0], 1, 1))
print(np.flip(np.append(np.flip(oup, axis=1), z, axis=1), axis=1))
#
# ze = np.zeros((_TOTAL_ARRAYS, _TIME2, _TIME1))
# cnt = 0
# oppp = inp.argsort(axis=1).flatten()
#
# print(inp)
# print(oup)
# print(oppp)
# print(ze)
# print(oppp.shape)
# for i in range(_TOTAL_ARRAYS):
# 	for j in range(_TIME2):
# 		z = oppp[cnt]
# 		print(i, j, z)
# 		ze[i, j, z] = 1.0
# 		cnt += 1
#
# print(ze)
