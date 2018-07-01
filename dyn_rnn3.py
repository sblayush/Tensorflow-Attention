import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

_EPOCHS = 10
_TOTAL_ARRAYS = 10
_BATCH_SIZE = 10
_TIME1 = 5
_TIME2 = 5
_ENC_N = 30
_DEC_N = 30


class DynRnn:
	def __init__(self):
		self.X = None
		self.Y = None
		self.Z = None
		self.op = None
		self.loss = None
		self.train = None
		self.alphas = None
		self.attn = None
		self.batch_size = None
		self.dec_outputs = None
		self.cell_dec = None
		self.is_running = None
		self.merge = None
		self.model()

	def _linear(
			self, inputs, n_output, name, bias_init=tf.zeros_initializer,
			weight_init=tf.random_normal_initializer, activation=None):
		assert inputs.get_shape()[-1] is not None
		with tf.variable_scope(name):
			weights = tf.get_variable('w', [inputs.get_shape()[-1], n_output], initializer=weight_init)
			bias = tf.get_variable('b', [n_output], initializer=bias_init)
			op = tf.matmul(inputs, weights) + bias

			if activation:
				op = activation(op)
		return op

	def _body(self, enc_outputs, time, dec_state, dec_out, dec_inp, alphas_arr):
		# dec_inp = tf.squeeze(self.Y[:, time:time + 1, :], axis=1)
		dec_inp = tf.cond(self.is_running, lambda: dec_inp, lambda: tf.squeeze(self.Y[:, time:time + 1, :], axis=1))
		dec_oup, dec_state = self.cell_dec(dec_inp, dec_state)
		dec_oup_attn = tf.expand_dims(dec_oup, axis=1)

		encoder_n = enc_outputs.shape[2].value  # D value - hidden size of the RNN layer encoder
		# encoder_t = encoder_states.shape[1].value  # D value - hidden size of the RNN layer encoder
		decoder_n = dec_oup_attn.shape[2].value  # D value - hidden size of the RNN layer decoder
		# batch_size = encoder_states.shape[0].value

		# Trainable parameters
		with tf.variable_scope("attn_weight"):
			w_omega = tf.get_variable(name="w_omega", shape=[encoder_n, decoder_n],
			                          initializer=tf.random_normal_initializer())  # [D1, D2]

		with tf.variable_scope("Score"):
			enc_reshape = tf.reshape(enc_outputs, [-1, encoder_n], name="enc_reshape")  # [(B*T1), D1]
			h1 = tf.matmul(enc_reshape, w_omega)  # [(B*T1), D1][D1, D2] = [(B*T1), D2]
			h1_reshape = tf.reshape(h1, tf.stack([self.batch_size, -1, decoder_n]), name="h1_reshape")  # [B, T1, D2]
			dec_transpose = tf.transpose(dec_oup_attn, [0, 2, 1])  # [B, D2, T2]
			# score = tf.matmul(encoder_states, dec_transpose)    # [B, T1, D2][B, D2, T2] = [B, T1, T2]
			score = tf.matmul(h1_reshape, dec_transpose)  # [B, T1, D2][B, D2, T2] = [B, T1, T2]
			score_transpose = tf.transpose(score, [0, 2, 1])  # [B, T2, T1]

		with tf.variable_scope("align"):
			# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
			alphas = tf.nn.softmax(score_transpose, axis=2, name='alphas')  # [B, T2, T1] with softmax on T1

		# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
		outputs_argmax = tf.argmax(alphas, axis=2, name="attn/outputs", output_type=tf.int32)  # [B, T2]
		print(outputs_argmax)
		dec_oup = tf.squeeze(tf.gather_nd(params=self.X, indices=index_matrix_to_pairs(outputs_argmax)), axis=1)
		dec_out = dec_out.write(time, dec_oup)
		alphas_arr = alphas_arr.write(time, outputs_argmax)
		return enc_outputs, time + 1, dec_state, dec_out, dec_oup, alphas_arr

	def model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='X')
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='Y')
		self.Z = tf.placeholder(dtype=tf.int32, shape=[None, None, 1], name='Z')
		self.is_running = tf.placeholder(dtype=tf.bool)
		# dec_inp = tf.constant(-1.0, dtype=tf.float32, shape=[1], name="dec_inp")

		z_squeeze = tf.squeeze(self.Z, axis=-1)
		z_onehot = tf.one_hot(z_squeeze, depth=_TIME1, axis=-1)
		self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")

		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=_ENC_N, name="enc_cell")
		self.cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=_DEC_N, name="dec_cell")

		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		# batch_size = enc_state.shape[0].value
		dec_inp = tf.ones_like(self.X[:, 0, :]) * -1
		dec_outputs_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		alphas_arr = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
		time = 0
		# for i in enc_outputs[:, :, :]:
		# 	self.dec_outputs = tf.concat([self.dec_outputs, enc_outputs[:, i, :]])
		dec_state = self.cell_dec.zero_state(self.batch_size, dtype=tf.float32)
		enc_outputs, time, dec_state, deco_arr, deco_inp, alphas_arr = self._body(enc_outputs, time, dec_state, dec_outputs_arr, dec_inp, alphas_arr)
		# self.dec_outputs = tf.squeeze(deco.stack(), axis=0)
		self.dec_outputs = deco_arr.stack()
		self.alphas = alphas_arr.stack()
		self.dec_outputs = tf.transpose(self.dec_outputs, [1, 0, 2])
		print(self.dec_outputs)
		# print(math_ops.to_int32(self.batch_size))
		# self.dec_outputs, dec_state = tf.nn.dynamic_rnn(cell=self.cell_dec, inputs=self.Y, dtype=tf.float32)

		# O = tf.get_variable('o', shape=[None, 5, 10], dtype=tf.float32)
		# O = tf.reduce_sum(enc_outputs, axis=1)

		# self.attn, self.alphas = self._attention(enc_outputs, self.dec_outputs)
		# self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=z_onehot, logits=self.attn))
		self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.dec_outputs, name="sq_diff"), name="loss")
		tf.summary.histogram("enc/kernel", tf.get_default_graph().get_tensor_by_name('rnn/enc_cell/kernel:0'))
		tf.summary.histogram("dec/kernel", tf.get_default_graph().get_tensor_by_name('dec_cell/kernel:0'))
		tf.summary.scalar("loss", self.loss)
		self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
		self.merge = tf.summary.merge_all()


def index_matrix_to_pairs(index_matrix):
	# [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
	#                        [[0, 2], [1, 3], [2, 1]]]
	replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
	rank = len(index_matrix.get_shape())
	if rank == 2:
		replicated_first_indices = tf.tile(
			tf.expand_dims(replicated_first_indices, dim=1),
			[1, tf.shape(index_matrix)[1]])
	return tf.stack([replicated_first_indices, index_matrix], axis=rank)


if __name__ == '__main__':
	my_rnn = DynRnn()

	inp = np.random.rand(_TOTAL_ARRAYS, _TIME1, 1)
	oup = np.sort(inp, axis=1)
	ze = np.zeros((_TOTAL_ARRAYS, _TIME2, _TIME1))
	cnt = 0
	op = inp.argsort(axis=1)
	# for i in range(_BATCH_SIZE):
	# 	for j in range(_TIME2):
	# 		ze[i, j, op[cnt]] = 1
	# 		cnt += 1

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	for i in range(_EPOCHS):
		start = cnt * _BATCH_SIZE
		end = (cnt + 1) * _BATCH_SIZE
		# att, lo, _ = sess.run(
		# 	[my_rnn.alphas, my_rnn.loss, my_rnn.train],
		# 	feed_dict={my_rnn.X: inp[start:end], my_rnn.Y: oup[start:end], my_rnn.Z: op[start:end], my_rnn.batch_size: _BATCH_SIZE})
		summ, alpa, deco, lo, yy, _ = sess.run(
			[my_rnn.merge, my_rnn.alphas, my_rnn.dec_outputs, my_rnn.loss, my_rnn.Y, my_rnn.train],
			feed_dict={my_rnn.X: inp[start:end], my_rnn.Y: oup[start:end], my_rnn.Z: op[start:end],
			           my_rnn.batch_size: _BATCH_SIZE, my_rnn.is_running: False})
		# print(deco)
		print(lo)
		# print(alpa.flatten())

		train_writer = tf.summary.FileWriter('/home/ayush/Desktop/pyCharm/summarization/summaries', sess.graph)
		train_writer.add_summary(summ, i)
		train_writer.flush()
	# cnt += 1
	# if cnt == _TOTAL_ARRAYS//_BATCH_SIZE:
	# 	cnt = 0
	print(np.asarray(deco)[-1:, :, :])
	# print(np.asarray(yy)[-1:, :, :])
	print(np.asarray(deco).shape)
	print(oup[-1:, :, :])
	print("#" * 8)
	# print(inp[-1:, :, :])
	# print(att)
	# print(att.argmax(axis=2).reshape(-1, _TIME1))
	# print(op.reshape(-1, _TIME1))

	alpa, deco = sess.run([my_rnn.alphas, my_rnn.dec_outputs],
	                      feed_dict={my_rnn.X: inp[0:2], my_rnn.Y: oup[0:2], my_rnn.batch_size: 2,
	                                 my_rnn.is_running: True})
	print(inp[0:2].flatten())
	print(oup[0:2].flatten())
	print(np.asarray(deco))
	print(alpa)
	print("#" * 8)

	inp = np.random.rand(1, _TIME1, 1)
	oup = np.sort(inp, axis=1)
	alpa, deco = sess.run([my_rnn.alphas, my_rnn.dec_outputs],
	                      feed_dict={my_rnn.X: inp, my_rnn.Y: oup, my_rnn.batch_size: 1, my_rnn.is_running: True})
	print(inp.flatten())
	print(oup.flatten())
	print(np.asarray(deco))
	print(alpa)
	print("#" * 8)

	print(tf.trainable_variables())
