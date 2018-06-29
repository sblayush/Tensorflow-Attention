import tensorflow as tf
import numpy as np


TIME_STEP = 100

class DynRnn():
	
	def __init__(self):
		self.X = None
		self.Y = None
		self.O = None
		self.attn = None
		self.loss = None
		self.train = None
		self.alphas = None
	
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
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='Y')
		self.OP = tf.placeholder(dtype=tf.int64, shape=[None, ], name='OP')
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, name="enc_cell")
		# cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, name="dec_cell")
		
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		# dec_outputs, dec_state = tf.nn.dynamic_rnn(cell=cell_dec, inputs=self.Y, dtype=tf.float32)
		# O = tf.get_variable('o', shape=[None, 5, 10], dtype=tf.float32)
		# self.O = tf.reduce_sum(enc_outputs, axis=1)
		
		# print(enc_outputs)
		# print(enc_state)
		
		hidden_size = enc_outputs.shape[2].value  # D value - hidden size of the RNN layer
		# Trainable parameters
		with tf.variable_scope("dec"):
			w_omega = tf.Variable(tf.random_normal([hidden_size, 5], stddev=0.1), name="w_omega")
			b_omega = tf.Variable(tf.random_normal([5], stddev=0.1), name="b_omega")
			u_omega = tf.Variable(tf.random_normal([5], stddev=0.1), name="u_omega")
		
		# with tf.name_scope('v'):
			# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
			#  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
			v = tf.tanh(tf.tensordot(enc_outputs, w_omega, axes=1, name="tensordot_1") + b_omega, name="my_tanh")
		
		# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
		vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
		self.alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
		
		# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
		attn = tf.reduce_sum(enc_outputs * tf.expand_dims(self.alphas, -1), 1, name="my_sum")
		
		# pos_argmax = tf.argmax(attn, axis=1, name="argmax_pos")         # [B, 1]
		# print(pos_argmax)
		# self.pos = tf.cast(pos_argmax, tf.float32, name="cast_pos")   # [B x 1]
		# print(self.pos)
		# diff = self.pos - self.OP
		one_hot_OP = tf.one_hot(self.OP, TIME_STEP, axis=-1)
		print(one_hot_OP)
		print(vu)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vu, labels=one_hot_OP))
		# diff = tf.squared_difference(self.alphas, one_hot_OP, name="Diff")
		# print(diff)
		# self.loss = tf.reduce_mean(diff, name="Loss")
		# self.loss = diff
		# print(self.loss)
		
		# for var in tf.trainable_variables():
		# 	print('{}--Shape: {}'.format(var.name, var.get_shape()))
		self.train = tf.train.AdamOptimizer(0.01).minimize(self.loss)
		
		return enc_outputs, enc_state
	

if __name__ == '__main__':
	my_rnn = DynRnn()
	
	inp = np.random.rand(1000, TIME_STEP, 1)
	oup = inp.argmax(axis=1).flatten()
	enc_outputs, enc_state = my_rnn.model()
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	# print(inp)
	# print(oup)
	# print(oup.shape)
	
	for i in range(100):
		_, lo, alph = sess.run([my_rnn.train, my_rnn.loss, my_rnn.alphas], feed_dict={my_rnn.X: inp, my_rnn.OP: oup})
		print(lo)
		# print(alph)
		# print(alph.argmax(axis=1))
	
	inp = np.random.rand(10, TIME_STEP, 1)
	oup = inp.argmax(axis=1).flatten()
	print(oup)
	alph = sess.run([my_rnn.alphas], feed_dict={my_rnn.X: inp, my_rnn.OP: oup})[0]
	print(alph.argmax(axis=1))
