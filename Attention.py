import tensorflow as tf


class Attention:
	def __init__(self):
		pass
	
	def _attention(self, encoder_states, attention_size, decoder_state=None, time_major=False, return_alphas=False):
		if isinstance(encoder_states, tuple):
			# In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
			encoder_states = tf.concat(encoder_states, 2)
		
		if time_major:
			# (T,B,D) => (B,T,D)
			encoder_states = tf.transpose(encoder_states, [1, 0, 2])
		
		hidden_size = encoder_states.shape[2].value     # D value - hidden size of the RNN layer
		batch_size = encoder_states.shape[0].value      # B value - hidden size of the RNN layer
		# Trainable parameters
		w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name="w_omega")
		b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name="b_omega")
		u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name="u_omega")
		
		with tf.variable_scope("Score"):
			# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
			#  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
			v = tf.tanh(tf.tensordot(encoder_states, w_omega, axes=1) + b_omega)
		
		# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
		vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
		alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
		
		# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
		output = tf.reduce_sum(encoder_states * tf.expand_dims(alphas, -1), 1)
		
		if not return_alphas:
			return output
		else:
			return output, alphas