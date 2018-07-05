import tensorflow as tf
import numpy as np
from Attention.Attention import Attention

_EPOCHS = 1250
_TOTAL_ARRAYS = 1000
_BATCH_SIZE = 10
_TIME1 = 20
_TIME2 = 1
_N_LSTM = 15
_DEPTH = 1
_LEARNING_RATE = 0.01


class MaxModel:
	def __init__(self):
		self.X = None
		self.Y = None
		self.score = None
		self.outputs = None
		self.loss = None
		self.train = None
		self._attention = Attention(activation=tf.tanh).attention
		self._model()
	
	def _model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, _DEPTH], name="X")      # [B, T1, D1]
		self.Y = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Y")              # [B, T1]
		
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=_N_LSTM)
		
		enc_outputs, enc_states = tf.nn.dynamic_rnn(cell=cell, inputs=self.X, dtype=tf.float32)
		enc_outputs_attn, score = self._attention(
			encoder_states=enc_outputs, enc_inp=self.X, return_score=True)    # [B, 1, T1], [B, 1, T1]
		
		self.score = tf.squeeze(score, axis=1)
		self.outputs = tf.squeeze(enc_outputs_attn, axis=1)
		
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=score))
		self.train = tf.train.AdamOptimizer(_LEARNING_RATE).minimize(self.loss)
	
	
if __name__ == '__main__':
	my_model = MaxModel()
	
	inp = np.random.rand(_TOTAL_ARRAYS, _TIME1, _DEPTH)
	oup = np.argmax(inp, axis=1)
	
	# print(inp)
	# print(op)
	
	ze = np.zeros((_TOTAL_ARRAYS, _TIME1))
	cnt = 0
	for i in range(_TOTAL_ARRAYS):
		ze[i, oup[cnt]] = 1
		cnt += 1
	
	# print(ze)
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	cnt = 0
	for i in range(_EPOCHS):
		start = cnt * _BATCH_SIZE
		end = (cnt + 1) * _BATCH_SIZE
		f_dict = {
			my_model.X: inp[start:end],
			my_model.Y: ze[start:end]
		}
		_, lo, sc, ou = sess.run([my_model.train, my_model.loss, my_model.score, my_model.outputs], feed_dict=f_dict)
		print(lo)
	print(inp[0:1, :, :])
	print(ze[0:1, :])
	print("#" * 25)
	
	f_dict = {
		my_model.X: inp[0:2]
	}
	deco = sess.run(my_model.outputs, feed_dict=f_dict)
	print(inp[0:2])
	print(np.asarray(deco))
	print("#" * 8)
	
	b_size = 2
	inp = np.random.rand(b_size, _TIME1*3, _DEPTH)
	f_dict = {
		my_model.X: inp
	}
	deco = sess.run(my_model.outputs, feed_dict=f_dict)
	print(inp)
	print(np.max(inp, axis=1))
	print(np.asarray(deco))
	print("#" * 8)
	
	b_size = 2
	inp = np.random.rand(b_size, _TIME1, _DEPTH)
	f_dict = {
		my_model.X: inp
	}
	deco = sess.run(my_model.outputs, feed_dict=f_dict)
	print(inp)
	print(np.max(inp, axis=1))
	print(np.asarray(deco))
	print("#" * 8)
