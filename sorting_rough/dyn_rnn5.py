import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops

_EPOCHS = 10000
_TOTAL_ARRAYS = 10000
_BATCH_SIZE = 10
_TIME1 = 20
_TIME2 = 20
_ENC_N = 15
_DEC_N = 15
_DEPTH = 1


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
		self.cell_dec = None
		self.dec_outputs = None
		self.scores = None
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

	def _body(self, enc_outputs, time, dec_state, dec_out_arr, dec_inp, score_arr):
		dec_inp = tf.cond(self.is_running, lambda: dec_inp, lambda: tf.squeeze(self.Y[:, time:time + 1, :], axis=1))
		dec_cell_oup, dec_state = self.cell_dec(dec_inp, dec_state)
		dec_oup_attn = tf.expand_dims(dec_cell_oup, axis=1, name="dec_oup_attn")
		dec_oup, score = _attention(self.X, enc_outputs, dec_oup_attn)
		dec_oup = tf.squeeze(dec_oup, axis=1, name="dec_oup")
		score = tf.squeeze(score, axis=1, name="score")
		dec_out_arr = dec_out_arr.write(time, dec_oup)
		score_arr = score_arr.write(time, score)
		return enc_outputs, time+1, dec_state, dec_out_arr, dec_oup, score_arr
	
	def model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, _DEPTH], name='X')
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, _DEPTH], name='Y')
		self.Z = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='Z')
		self.is_running = tf.placeholder(dtype=tf.bool)
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=_ENC_N, name="enc_cell")
		self.cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=_DEC_N, name="dec_cell")
		
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		
		dec_inp = tf.ones_like(self.X[:, 0, :]) * -1
		dec_outputs_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		score_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		time = tf.constant(0, dtype=tf.int32)
		batch_size = array_ops.shape(enc_outputs)[0]
		max_time = array_ops.shape(self.Y)[1]
		# flat_input = nest.flatten(self.Y)
		# max_time = array_ops.shape(flat_input[0])[0]
		dec_state = self.cell_dec.zero_state(batch_size, dtype=tf.float32)
		# dec_state = enc_state
		enc_outputs, time, dec_state, deco_arr, deco_inp, score_arr = tf.while_loop(
			lambda e, t, *_: t < max_time,
			self._body,
			loop_vars=[enc_outputs, time, dec_state, dec_outputs_arr, dec_inp, score_arr])
		
		self.scores = tf.transpose(score_arr.stack(), [1, 0, 2], name="scores")    # [B, T2+1, T1]
		self.dec_outputs = tf.transpose(deco_arr.stack(), [1, 0, 2], name="dec_outputs")    #[B, T2+1, _DEPTH]
		
		# Without body(while_loop)
		# dec_outputs, _dec_states = tf.nn.dynamic_rnn(cell=cell_dec, inputs=self.Y, dtype=tf.float32)
		# self.dec_outputs, self.score_transpose = self._attention(enc_outputs, dec_outputs)
		
		# Discarding last _TIME in outputs_argmax as it is end
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Z, logits=self.scores[:, :-1, :]), name="loss")
		tf.summary.scalar("loss", self.loss)
		optimizer = tf.train.AdamOptimizer(0.01, name="optimizer")
		self.train = optimizer.minimize(self.loss, name="train")
		self.merge = tf.summary.merge_all()
	
	
def _attention(X, encoder_states, decoder_states, time_major=False, return_alphas=True):
	if isinstance(encoder_states, tuple):
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
		encoder_states = tf.concat(encoder_states, 2)

	if time_major:
		# (T,B,D) => (B,T,D)
		encoder_states = tf.transpose(encoder_states, [1, 0, 2])
		decoder_states = tf.transpose(decoder_states, [1, 0, 2])

	encoder_n = encoder_states.shape[2].value  # D value - hidden size of the RNN layer encoder
	decoder_n = decoder_states.shape[2].value  # D value - hidden size of the RNN layer decoder
	batch_size = array_ops.shape(decoder_states)[0]

	# Trainable parameters
	with tf.variable_scope("attn/attn_weight"):
		w_omega = tf.get_variable(name="w_omega", shape=[encoder_n, decoder_n], initializer=tf.random_normal_initializer())  # [D1, D2]

	with tf.variable_scope("attn/Score"):
		enc_reshape = tf.reshape(encoder_states, [-1, encoder_n], name="enc_reshape")   # [(B*T1), D1]
		h1 = tf.matmul(enc_reshape, w_omega)    # [(B*T1), D1][D1, D2] = [(B*T1), D2]
		h1_reshape = tf.reshape(h1, tf.stack([batch_size, -1, decoder_n]), name="h1_reshape")     # [B, T1, D2]
		dec_transpose = tf.transpose(decoder_states, [0, 2, 1]) # [B, D2, T2]
		score = tf.matmul(h1_reshape, dec_transpose)   # [B, T1, D2][B, D2, T2] = [B, T1, T2]
		score_transpose = tf.transpose(score, [0, 2, 1])    # [B, T2, T1]

	with tf.variable_scope("attn/align"):
		# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
		alphas = tf.nn.softmax(score_transpose, axis=2, name='alphas')  # [B, T2, T1] with softmax on T1

	# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
	with tf.variable_scope("attn/outputs"):
		outputs_argmax = tf.argmax(alphas, axis=2, name="outputs_argmax", output_type=tf.int32)  # [B, T2]
		outputs = tf.gather_nd(params=X, indices=index_matrix_to_pairs(outputs_argmax))
		# outputs = tf.reduce_sum(tf.matmul(alphas, encoder_states), 1)
	
	if not return_alphas:
		return outputs
	else:
		return outputs, score_transpose
	
	
def index_matrix_to_pairs(index_matrix):
	# [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
	#                        [[0, 2], [1, 3], [2, 1]]]
	with tf.variable_scope("index_matrix_to_pairs"):
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
	
	z = np.ones((oup.shape[0], 1, _DEPTH))*-1
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
	print("#"*8)
	
	b_size = 1
	inp = np.random.rand(b_size, _TIME1//3, _DEPTH)
	oup = np.sort(inp, axis=1)
	f_dict = {
		my_rnn.X: inp,
		my_rnn.Y: np.zeros_like(inp),
		my_rnn.is_running: True}
	deco = sess.run(my_rnn.dec_outputs, feed_dict=f_dict)
	print(inp)
	print(oup)
	print(np.asarray(deco))
	print("#"*8)

	print(tf.trainable_variables())
