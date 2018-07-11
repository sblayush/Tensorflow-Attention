import tensorflow as tf
from tensorflow.python.ops import array_ops


class Attention:
	"""
	Attention class to implement attntion mentioned in papers:
		https://arxiv.org/abs/1506.03134
		https://arxiv.org/abs/1704.04368
		https://arxiv.org/abs/1508.04025
	"""
	
	def __init__(self, activation=None, is_decoder_present=False):
		self.activation = activation
		self.is_decoder_present = is_decoder_present
	
	def attention(
			self,
			encoder_states,
			enc_inp=None,
			decoder_states=None,
			time_major=False,
			return_score=False,
			return_context=False,
			return_alphas=False
			):
		"""
		score(t) = h(s).W.trans(h(t))  dim:[B, T2, T1] = [B, T1, D1].[D1, D2].transpose([B, T2, D2])
		alphas = softmax(t)
		Args:
			:param encoder_states: Encoder outputs on which attention is to be applied, shape: [B, T1, D1]
			:param enc_inp: Encoder inputs, shape: [B, T1, Dim]
			:param decoder_states: Decoder states, shape: [B, T2, D2]
				if None: self attention is implemented with dec_states tf.ones([B, 1, 1])
			:param activation: activation function to apply on score
			:param time_major: Default is False
				True if shape of encoder/decoder is (T, B, D)
				Flase if shape of encoder is (B, T, D)
			:param return_score: Default is False
			:param return_context: Defaults to False
		Returns:
			:return:
				returns outputs([B, T2, D1]) and
				if return_score is True: scores ([B, T2, T1])
				if return_context is True: context ([B, D1])
		"""
		if isinstance(encoder_states, tuple):
			# In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
			encoder_states = tf.concat(encoder_states, 2)
		
		if time_major:
			# (T,B,D) => (B,T,D)
			encoder_states = tf.transpose(encoder_states, [1, 0, 2])
			decoder_states = tf.transpose(decoder_states, [1, 0, 2])

		batch_size = array_ops.shape(encoder_states)[0]
		
		if not self.is_decoder_present:
			decoder_states = tf.ones([batch_size, 1, 1], dtype=encoder_states.dtype, name="dec_states")
			
		encoder_n = encoder_states.shape[2].value  # D1 value - hidden size of the RNN layer encoder
		decoder_n = decoder_states.shape[2].value  # D2 value - hidden size of the RNN layer decoder
		
		# Trainable parameters
		with tf.variable_scope("attn/attn_weight"):
			w_omega = tf.get_variable(
				name="w_omega", shape=[encoder_n, decoder_n], initializer=tf.random_normal_initializer())  # [D1, D2]
		
		with tf.variable_scope("attn/score"):
			enc_reshape = tf.reshape(encoder_states, [-1, encoder_n], name="enc_reshape")  # [(B*T1), D1]
			h1 = tf.matmul(enc_reshape, w_omega)  # [(B*T1), D1][D1, D2] = [(B*T1), D2]
			h1_reshape = tf.reshape(h1, tf.stack([batch_size, -1, decoder_n]), name="h1_reshape")  # [B, T1, D2]
			dec_transpose = tf.transpose(decoder_states, [0, 2, 1])  # [B, D2, T2]
			if not self.activation:
				score = tf.matmul(h1_reshape, dec_transpose)  # [B, T1, D2][B, D2, T2] = [B, T1, T2]
			else:
				score = self.activation(tf.matmul(h1_reshape, dec_transpose))  # [B, T1, D2][B, D2, T2] = [B, T1, T2]
			score_transpose = tf.transpose(score, [0, 2, 1])  # [B, T2, T1]
		
		with tf.variable_scope("attn/align"):
			alphas = tf.nn.softmax(score_transpose, axis=2, name='alphas')  # [B, T2, T1] with softmax on T1
		
		with tf.variable_scope("attn/outputs"):
			alphas_argmax = tf.argmax(alphas, axis=2, name="outputs_argmax", output_type=tf.int32)  # [B, T2]
			outputs = tf.gather_nd(params=enc_inp, indices=self._index_matrix_to_pairs(alphas_argmax))

		# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D1) shape
		with tf.variable_scope("attn/context_vec"):
			context = tf.reduce_sum(tf.matmul(alphas, encoder_states), axis=1, name="context")   # [B, D1]
		ret = [outputs]
		if return_score:
			ret.append(score_transpose)
		if return_context:
			ret.append(context)
		if return_alphas:
			ret.append(alphas)
		return tuple(ret)
		
	def _index_matrix_to_pairs(self, index_matrix):
		""" Method to create index to pairs in the first dimension
		[[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
								[[0, 2], [1, 3], [2, 1]]]
		Args:
		:param index_matrix: a Tensor of indexes
		Returns:
		:return Tensor representing the pairwise indexed index_matrix
		"""
		with tf.variable_scope("index_matrix_to_pairs"):
			replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
			rank = len(index_matrix.get_shape())
			if rank == 2:
				replicated_first_indices = tf.tile(
					tf.expand_dims(replicated_first_indices, dim=1),
					[1, tf.shape(index_matrix)[1]])
			return tf.stack([replicated_first_indices, index_matrix], axis=rank)
