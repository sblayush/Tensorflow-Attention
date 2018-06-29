import tensorflow as tf
import numpy as np


class DynRnn():
	
	def __init__(self):
		pass
	
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
		X = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='X')
		Y = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='Y')
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, name="enc_cell")
		cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, name="dec_cell")
		
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=X, dtype=tf.float32)
		dec_outputs, dec_state = tf.nn.dynamic_rnn(cell=cell_dec, inputs=Y, dtype=tf.float32)
		
		# O = tf.get_variable('o', shape=[None, 5, 10], dtype=tf.float32)
		
		O = tf.reduce_sum(enc_outputs, axis=1)
		
		return enc_outputs, enc_state, X, Y, O
	
	
if __name__ == '__main__':
	my_rnn = DynRnn()
	
	inp = np.random.rand(2, 5, 2)
	
	enc_outputs, enc_state, X, Y, O = my_rnn.model()
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	oo, o, s = sess.run([O, enc_outputs, enc_state], feed_dict={X: inp})
	print(oo)
	print(o)
	print(s)