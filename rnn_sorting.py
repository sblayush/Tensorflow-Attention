import tensorflow as tf
import numpy as np

_TOTAL_ARRAYS = 1000
_BATCH_SIZE = 10
_TIME1 = 20
_TIME2 = 20
_ENC_N = 30
_DEC_N = 30
_LEARNING_RATE = 0.01


class RNNSorting:
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
	
	def model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='X')
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='Y')
		self.Z = tf.placeholder(dtype=tf.int32, shape=[None, None, 1], name='Z')
		z_squeeze = tf.squeeze(self.Z, axis=-1)
		z_onehot = tf.one_hot(z_squeeze, depth=_TIME1, axis=-1)
		self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=_ENC_N, name="enc_cell")
		cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=_DEC_N, name="dec_cell")
		
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		dec_outputs, dec_state = tf.nn.dynamic_rnn(cell=cell_dec, inputs=self.Y, dtype=tf.float32)
		
		# dec_state = cell_dec.zero_state(self.batch_size, tf.float32)
		# dec_outputs = []
		# for i in range(tf.shape(enc_outputs)[0].value):
		# 	dec_output, dec_state = tf.nn.dynamic_rnn(cell=cell_dec, inputs=self.Y[:, i, :], dtype=tf.float32, initial_state=dec_state)
		# 	dec_outputs.append(dec_output)
		
		# O = tf.get_variable('o', shape=[None, 5, 10], dtype=tf.float32)
		# O = tf.reduce_sum(enc_outputs, axis=1)
		
		self.attn, self.alphas = self._attention(enc_outputs, tf.stack(dec_outputs))
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=z_onehot, logits=self.attn))
		
		self.train = tf.train.AdamOptimizer(_LEARNING_RATE).minimize(self.loss)
	
	def _attention(self, encoder_states, decoder_states, time_major=False, return_alphas=True):
		if isinstance(encoder_states, tuple):
			# In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
			encoder_states = tf.concat(encoder_states, 2)
		
		if time_major:
			# (T,B,D) => (B,T,D)
			encoder_states = tf.transpose(encoder_states, [1, 0, 2])
			decoder_states = tf.transpose(decoder_states, [1, 0, 2])
		
		encoder_n = encoder_states.shape[2].value  # D value - hidden size of the RNN layer encoder
		# encoder_t = encoder_states.shape[1].value  # D value - hidden size of the RNN layer encoder
		decoder_n = decoder_states.shape[2].value  # D value - hidden size of the RNN layer decoder
		# batch_size = encoder_states.shape[0].value
		
		# Trainable parameters
		w_omega = tf.Variable(tf.random_normal([encoder_n, decoder_n], stddev=0.1), name="w_omega")  # [D1, D2]
		
		with tf.variable_scope("Score"):
			enc_reshape = tf.reshape(encoder_states, [-1, encoder_n], name="enc_reshape")  # [(B*T1), D1]
			h1 = tf.matmul(enc_reshape, w_omega)  # [(B*T1), D1][D1, D2] = [(B*T1), D2]
			h1_reshape = tf.reshape(h1, tf.stack([self.batch_size, -1, decoder_n]), name="h1_reshape")  # [B, T1, D1]
			dec_transpose = tf.transpose(decoder_states, [0, 2, 1])  # [B, D2, T2]
			score = tf.matmul(h1_reshape, dec_transpose)  # [B, T1, D1][B, D2, T2] = [B, T1, T2]
			score_transpose = tf.transpose(score, [0, 2, 1])  # [B, T1, T2]
		
		with tf.variable_scope("align"):
			# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
			alphas = tf.nn.softmax(score_transpose, axis=2, name='alphas')  # [B, T2, T1] with softmax on T1
		
		# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
		# outputs = tf.reduce_sum(encoder_states * tf.expand_dims(alphas, -1), 1)
		
		if not return_alphas:
			return score_transpose
		else:
			return score_transpose, alphas


if __name__ == '__main__':
	my_rnn = RNNSorting()
	
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
	
	for i in range(100000):
		start = cnt * _BATCH_SIZE
		end = (cnt + 1) * _BATCH_SIZE
		att, lo, _ = sess.run(
			[my_rnn.alphas, my_rnn.loss, my_rnn.train],
			feed_dict={my_rnn.X: inp[start:end], my_rnn.Y: oup[start:end], my_rnn.Z: op[start:end],
			           my_rnn.batch_size: _BATCH_SIZE})
		print(lo)
		cnt += 1
		if cnt == _TOTAL_ARRAYS // _BATCH_SIZE:
			cnt = 0
	
	# print(inp)
	# print(att)
	print(att.argmax(axis=2).reshape(-1, _TIME1))
	print(op.reshape(-1, _TIME1))

	while True:
		inp = np.asarray([list(map(float, input("Enter array string:").split()))]).reshape(1, -1, 1)
		att = sess.run(my_rnn.alphas, feed_dict={my_rnn.X: inp, my_rnn.Y: np.sort(inp, axis=1), my_rnn.batch_size: 1})
		print(att.argmax(axis=2))
		print(inp)
