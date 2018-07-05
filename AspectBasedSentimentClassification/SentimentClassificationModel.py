from Attention.Attention import Attention
from Linear.Linear import Linear
import tensorflow as tf
from tensorflow.python.ops import array_ops


class SentimentClassificationModel:
	"""
	Tensorflow implementation of Aspect based Sentiment Classification based on paper:
		https://aclweb.org/anthology/D16-1058
	SemEval dataset is use for training the model
	"""
	
	def __init__(self, n_inp, n_enc_h, n_aspect_class, n_polarity, n_aspect_embed, learning_rate=0.01):
		"""
		:param n_enc_h: int type: length of encoder features
		:param n_aspect_class:  int type: length of decoder features
		:param n_inp:  int type: length of input/output features
		:param n_polarity:
		:param n_aspect_embed:
		"""
		self.n_enc_h = n_enc_h
		self.n_aspect_class = n_aspect_class
		self.n_inp = n_inp
		self.n_polarity = n_polarity
		self.n_aspect_embed = n_aspect_embed
		self.learning_rate= learning_rate
		self.X = None
		self.AspectClass = None
		self.Y = None
		self.op = None
		self.loss = None
		self.train = None
		self.alphas = None
		self.is_running = None
		self.merge = None
		self.pred_class_softmax = None
		self._attention = Attention().attention
		self.linear = Linear().linear
		self._model()
		
	def _model(self):
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.n_inp], name='X')  # [B, T, D1]
		self.AspectClass = tf.placeholder(dtype=tf.float32, shape=[None, self.n_aspect_class], name='AspectClass')
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_polarity], name='Y')
		# self.is_running = tf.placeholder(dtype=tf.bool)
		
		batch_size = array_ops.shape(self.X)[0]
		time_steps = array_ops.shape(self.X)[1]
		
		cell_enc = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_enc_h, name="enc_cell")
		enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=cell_enc, inputs=self.X, dtype=tf.float32)
		
		aspect_embedding = self.linear(
			inputs=self.AspectClass,
			n_output=self.n_aspect_embed,
			name="aspect_embedding",
			activation=tf.tanh
		)
		
		tiled_aspect = tf.tile(tf.expand_dims(aspect_embedding, axis=1), [1, time_steps, 1])
		enc_op = tf.tanh(tf.concat([enc_outputs, tiled_aspect], axis=2))

		enc_outputs_attn, context, self.alphas = self._attention(
			encoder_states=enc_op, enc_inp=self.X, return_context=True, return_alphas=True)  # [B, D1]

		w_context = tf.get_variable(name='w_context', shape=[self.n_enc_h+self.n_aspect_embed, self.n_inp], dtype=tf.float32)
		w_state = tf.get_variable(name='w_state', shape=[self.n_enc_h, self.n_inp], dtype=tf.float32)
		
		# context = tf.squeeze(enc_op[:, -1:, :], axis=1)
		h_star = tf.tanh(tf.matmul(context, w_context)+tf.matmul(enc_state.h, w_state))
		
		pred_class = self.linear(
			inputs=h_star,
			n_output=self.n_polarity,
			name="pred_class",
		)
		
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=pred_class))
		self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		self.pred_class_softmax = tf.nn.softmax(pred_class, axis=1)
		
		tf.summary.scalar("loss", self.loss)
		self.merge = tf.summary.merge_all()
		
		print('Model created')
