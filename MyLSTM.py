import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
import numpy as np

TIME_STEPS = 10
N_UNITS = 30
N_INPUTS = 1


class MyLSTM(rnn_cell_impl.LayerRNNCell):
	def __init__(self, n_units):
		super().__init__()
		self.n_units = n_units
	
	def _linear(self, inp):
		n_inputs = inp.get_shape()[1]
		weights = tf.get_variable('W', [n_inputs, 4*self.n_units], dtype=inp.dtype)
		bias = tf.get_variable('B', [4*self.n_units], dtype=inp.dtype)
		return tf.nn.sigmoid(tf.matmul(inp, weights, name="cell_mul") + bias)

	def call(self, inputs, state):
		c, h = state
		gate_inputs = self._linear(tf.concat([inputs, h], axis=1))
		f, i, j, o = array_ops.split(gate_inputs, num_or_size_splits=4, axis=1)
		ct = tf.sigmoid(f)*c + tf.sigmoid(i)*tf.tanh(j) + c
		ht = tf.sigmoid(o) * tf.tanh(ct)
		return ct, ht
		
		
def LSTM_fwd(cell, inputs, state):
	states = []
	
	weights = tf.get_variable('W_lin', [N_UNITS, N_INPUTS], dtype=inp.dtype)
	bias = tf.get_variable('B_lin', [N_INPUTS], dtype=inp.dtype)
	
	def _linear(inp):
		return tf.nn.tanh(tf.matmul(inp, weights, name="fwd_mul") + bias)
	
	for time_step in range(TIME_STEPS):
		cell_inp = inputs[:, time_step:time_step+1, :]      # [B x 1 x N_INPUTS]
		cell_inp = tf.squeeze(cell_inp, axis=1)             # [B x N_INPUTS]
		state = cell(cell_inp, state)                       # ([B x N_UNITS], [B x N_UNITS])
		states.append(_linear(state[1]))                    # [B x N_INPUTS]
	return states
	
	
class model():
	def __init__(self, n_inps, n_units):
		self.n_inps = n_inps
		self.n_units = n_units
		
	def forward(self):
		X = tf.placeholder(tf.float64, [None, TIME_STEPS, self.n_inps], name='X')
		Y = tf.placeholder(tf.float64, [None, TIME_STEPS, self.n_inps], name='Y')
		
		state = (np.zeros([2, N_UNITS], dtype=np.float), np.zeros([2, self.n_units], dtype=np.float))
	
		ce = MyLSTM(self.n_units)
		states = LSTM_fwd(ce, X, state)
		states = tf.transpose(states, [1, 0, 2])
		loss = tf.reduce_mean(tf.squared_difference(states, X))
		optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
		return X, state, loss, optimize, states
	
		
if __name__ == '__main__':
	inp = np.random.rand(10, TIME_STEPS, N_INPUTS)
	my_model = model(N_INPUTS, N_UNITS)
	X, state, loss, optimize, states = my_model.forward()
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	st = None
	for _ in range(500):
		total_loss = 0
		st = []
		for i in range(5):
			my_loss, my_optimize, my_states = sess.run([loss, optimize, states], feed_dict={X: inp[2*i:2*(i+1), :, :]})
			total_loss += my_loss
			st += my_states.tolist()
		print(total_loss)
	cnt = 0
	st = np.asarray(st).ravel()
	for i in inp.ravel():
		print(i, st[cnt])
		cnt += 1
	print(inp.ravel())
	print(st)
