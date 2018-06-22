import tensorflow as tf
import numpy as np


class Model:
	def __init__(self):
		self.optimizer = tf.train.AdamOptimizer
		
	def _linear(self, inputs, n_output, name, bias_init=tf.zeros_initializer,
				weight_init=tf.random_normal_initializer, activation=None):
		
		assert inputs.get_shape()[-1] is not None
		with tf.variable_scope(name):
			weights = tf.get_variable('w', [inputs.get_shape()[-1], n_output], initializer=weight_init)
			bias = tf.get_variable('b', [n_output], initializer=bias_init)
			op = tf.matmul(inputs, weights) + bias
			
			if activation:
				op = activation(op)
		return op
	
	def encoder(self):
	
	
	def decoder(self):
		pass
	
	def evaluate(self):
		n_hidden1 = 5
		n_hidden2 = 5
		n_op = 2
		
		X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='X')
		Y = tf.placeholder(dtype=tf.float32, shape=[None, n_op], name='Y')
		
		h1 = self._linear(X, n_hidden1, name="h1", activation=tf.tanh)
		h2 = self._linear(h1, n_hidden2, name="h2", activation=tf.tanh)
		output = self._linear(h2, n_op, name="op")
		
		loss = tf.reduce_mean(tf.squared_difference(output, Y))
		train = self.optimizer(0.01).minimize(loss)
		
		return X, Y, output, loss, train
	

def evaluate_ann():
	my_model = Model()
	x, y, op, los, trai = my_model.evaluate()
	
	inp = [[0, 0], [0, 1], [1, 0], [1, 1]]
	opp = [[0, 0], [1, 0], [1, 0], [1, 1]]
	
	print(inp, opp)
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	for i in range(50):
		out, _, cost = sess.run([op, trai, los], feed_dict={x: inp, y: opp})
		print(out, cost)


def evaluate():
	# inp = "This is the sentence one."
	# op = "This is the expected sentence."
	
	inp = [
		[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]]
	op = [
		[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]
	
	

if __name__ == '__main__':
	evaluate()
	# inp = int_to_bin_list(4, 2)
	# opp = int_to_bin_list(4, 2)
	#
	# print(inp, opp)
