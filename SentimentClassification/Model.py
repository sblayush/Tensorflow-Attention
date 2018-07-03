from Attention.Attention import Attention
import tensorflow as tf
from tensorflow.python.ops import array_ops


class SortingModel:
	"""
	Tensorflow implementation of array sorting
	"""
	
	def __init__(self, n_inp, n_enc, n_dec):
		"""
		:param n_enc: int type: length of encoder features
		:param n_dec:  int type: length of decoder features
		:param n_inp:  int type: length of input/output features
		"""
		self.n_enc = n_enc
		self.n_dec = n_dec
		self.n_inp = n_inp
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
		self._attention = Attention().attention
		self._model()
		
	def _body(self, enc_outputs, time, dec_state, dec_out_arr, dec_inp, score_arr):
		dec_inp = tf.cond(self.is_running, lambda: dec_inp, lambda: tf.squeeze(self.Y[:, time:time + 1, :], axis=1))
		dec_cell_oup, dec_state = self.cell_dec(dec_inp, dec_state)
		dec_oup_attn = tf.expand_dims(dec_cell_oup, axis=1, name="dec_oup_attn")
		dec_oup, score = self._attention(self.X, enc_outputs, dec_oup_attn)
		dec_oup = tf.squeeze(dec_oup, axis=1, name="dec_oup")
		score = tf.squeeze(score, axis=1, name="score")
		dec_out_arr = dec_out_arr.write(time, dec_oup)
		score_arr = score_arr.write(time, score)
		return enc_outputs, time + 1, dec_state, dec_out_arr, dec_oup, score_arr
	
	def _model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.n_inp], name='X')
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, None, self.n_inp], name='Y')
		self.Z = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='Z')
		self.is_running = tf.placeholder(dtype=tf.bool)
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_enc, name="enc_cell")
		self.cell_dec = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_dec, name="dec_cell")
		
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		
		dec_inp = tf.ones_like(self.X[:, 0, :]) * -1
		dec_outputs_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		score_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		time = tf.constant(0, dtype=tf.int32)
		batch_size = array_ops.shape(enc_outputs)[0]
		max_time = array_ops.shape(self.Y)[1]
		dec_state = self.cell_dec.zero_state(batch_size, dtype=tf.float32)
		# dec_state = enc_state
		enc_outputs, time, dec_state, deco_arr, deco_inp, score_arr = tf.while_loop(
			lambda e, t, *_: t < max_time,
			self._body,
			loop_vars=[enc_outputs, time, dec_state, dec_outputs_arr, dec_inp, score_arr])
		
		self.scores = tf.transpose(score_arr.stack(), [1, 0, 2], name="scores")  # [B, T2+1, T1]
		self.dec_outputs = tf.transpose(deco_arr.stack(), [1, 0, 2], name="dec_outputs")  # [B, T2+1, _DEPTH]
		
		# Without body(while_loop)
		# dec_outputs, _dec_states = tf.nn.dynamic_rnn(cell=cell_dec, inputs=self.Y, dtype=tf.float32)
		# self.dec_outputs, self.score_transpose = self._attention(enc_outputs, dec_outputs)
		
		# Discarding last _TIME in outputs_argmax as it is end
		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Z, logits=self.scores[:, :-1, :]), name="loss")
		tf.summary.scalar("loss", self.loss)
		optimizer = tf.train.AdamOptimizer(0.01, name="optimizer")
		self.train = optimizer.minimize(self.loss, name="train")
		self.merge = tf.summary.merge_all()
