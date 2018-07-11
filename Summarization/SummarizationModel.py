from Attention.Attention import Attention
from Linear.Linear import Linear
import tensorflow as tf
from tensorflow.python.ops import array_ops


class SummarizationModel:
	"""
	Tensorflow implementation of Aspect based Sentiment Classification based on paper:
		https://aclweb.org/anthology/D16-1058
	SemEval dataset is use for training the model
	"""
	
	def __init__(self, n_inp, n_enc_h, n_dec_h, learning_rate=0.01):
		"""
		:param n_enc_h: int type: length of encoder features
		:param n_inp:  int type: length of input/output features
		"""
		self.n_enc_h = n_enc_h
		self.n_dec_h = n_dec_h
		self.n_inp = n_inp
		self.cell_dec = None
		self.learning_rate= learning_rate
		self.X = None
		self.Y = None
		self.op = None
		self.loss = None
		self.train = None
		self.is_running = None
		self.context_arr = None
		self.merge = None
		self._attention = Attention(is_decoder_present=True).attention
		self.linear = Linear().linear
		self._model()
	
	def _body(self, enc_outputs, time, dec_state, dec_out_arr, dec_inp, context_arr):
		dec_inp = tf.cond(self.is_running, lambda: dec_inp, lambda: tf.squeeze(self.Y[:, time:time + 1, :], axis=1))
		dec_cell_oup, dec_state = self.cell_dec(dec_inp, dec_state)
		dec_oup_attn = tf.expand_dims(dec_cell_oup, axis=1, name="dec_oup_attn")
		dec_oup, context = self._attention(enc_outputs, self.X, dec_oup_attn, return_context=True)  # [B, 1, D2], [B, D2]
		context = self.linear(
			inputs=context,
			n_output=self.n_inp,
			name="out_embedding"
		)
		dec_oup = tf.squeeze(dec_oup, axis=1, name="dec_oup")
		dec_out_arr = dec_out_arr.write(time, dec_oup)
		context_arr = context_arr.write(time, context)
		return enc_outputs, time + 1, dec_state, dec_out_arr, dec_oup, context_arr
	
	def _model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.n_inp], name='X')  # [B, T1, D1]
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, self.n_inp], name='Y')  # [B, T2, D1]
		self.is_running = tf.placeholder(dtype=tf.bool)
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_enc_h, name="enc_cell")
		self.cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_dec_h, name="dec_cell")
		
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		
		dec_inp = tf.ones_like(self.X[:, 0, :]) * -1
		dec_outputs_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		context_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		time = tf.constant(0, dtype=tf.int32)
		batch_size = array_ops.shape(enc_outputs)[0]
		max_time = array_ops.shape(self.Y)[1]
		dec_state = self.cell_dec.zero_state(batch_size, dtype=tf.float32)
		
		enc_outputs, time, dec_state, deco_arr, deco_inp, context_arr = tf.while_loop(
			lambda e, t, *_: t < max_time,
			self._body,
			loop_vars=[enc_outputs, time, dec_state, dec_outputs_arr, dec_inp, context_arr])
		
		self.context_arr = tf.transpose(context_arr.stack(), [1, 0, 2], name="context_transpose")
		
		self.loss = tf.reduce_mean(tf.squared_difference(self.Y[:, 1:, :], self.context_arr[:, :-1, :]))
		self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		tf.summary.scalar("loss", self.loss)
		self.merge = tf.summary.merge_all()
		
		print('Model created')
