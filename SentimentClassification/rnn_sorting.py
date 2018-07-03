import numpy as np
import tensorflow as tf
from sorting.Model import SortingModel

_EPOCHS = 10000
_TOTAL_ARRAYS = 10000
_BATCH_SIZE = 10
_TIME1 = 20
_TIME2 = 20
_ENC_N = 15
_DEC_N = 15
_DEPTH = 1


if __name__ == '__main__':
	my_rnn = SortingModel(n_inp=_DEPTH, n_enc=_ENC_N, n_dec=_DEC_N)
	inp = np.random.rand(_TOTAL_ARRAYS, _TIME1, 1)
	oup = np.sort(inp, axis=1)
	
	z = np.ones((oup.shape[0], 1, _DEPTH)) * -1
	oup = np.flip(np.append(np.flip(oup, axis=1), z, axis=1), axis=1)
	
	ze = np.zeros((_TOTAL_ARRAYS, _TIME2, _TIME1))
	cnt = 0
	oppp = inp.argsort(axis=1).flatten()
	for i in range(_TOTAL_ARRAYS):
		for j in range(_TIME2):
			ze[i, j, oppp[cnt]] = 1
			cnt += 1
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	cnt = 0
	for i in range(_EPOCHS):
		start = cnt * _BATCH_SIZE
		end = (cnt + 1) * _BATCH_SIZE
		f_dict = {
			my_rnn.X: inp[start:end],
			my_rnn.Y: oup[start:end],
			my_rnn.Z: ze[start:end],
			my_rnn.is_running: False
		}
		score_trans, lo, _ = sess.run([my_rnn.scores, my_rnn.loss, my_rnn.train], feed_dict=f_dict)
		print(lo)
	print(inp[0:1, :, :])
	print(oup[0:1, :, :])
	print(ze[0:1, :, :])
	print(score_trans[0:1, :, :])
	print("#" * 25)
	
	f_dict = {
		my_rnn.X: inp[0:2],
		my_rnn.Y: np.zeros_like(inp[0:2]),
		my_rnn.is_running: True}
	deco = sess.run(my_rnn.dec_outputs, feed_dict=f_dict)
	print(inp[0:2])
	print(oup[0:2])
	print(np.asarray(deco))
	print("#" * 8)
	
	b_size = 1
	inp = np.random.rand(b_size, _TIME1 // 3, _DEPTH)
	oup = np.sort(inp, axis=1)
	f_dict = {
		my_rnn.X: inp,
		my_rnn.Y: np.zeros_like(inp),
		my_rnn.is_running: True}
	deco = sess.run(my_rnn.dec_outputs, feed_dict=f_dict)
	print(inp)
	print(oup)
	print(np.asarray(deco))
	print("#" * 8)
	
	print(tf.trainable_variables())
