import tensorflow as tf
from datetime import datetime as dt
from tensorflow.python.ops import array_ops
from src.main.models.Model import Model
from src.main.utility.get_logger import logger


class OFTModel1(Model):
	def __init__(self, **kwargs):
		super().__init__()
		self.lstm_width_enc = kwargs['lstm_width_enc']  # L
		self.input_dimensions = kwargs['input_dimensions']  # S
		self.part3_hidden_1 = kwargs['part3_hidden_1']  # D3_1
		self.part3_hidden_2 = kwargs['part3_hidden_2']  # D3_2
		self.part5_hidden_1 = kwargs['part5_hidden_1']  # D5_ 1
		self.part5_hidden_2 = kwargs['part5_hidden_2']  # D5_2
		self.optimizer = kwargs['optimizer']
		self.n_op_part3 = kwargs['n_op_part3']  # C3
		self.n_op_part5 = kwargs['n_op_part5']  # C5
		self.reg_scale = kwargs['reg_scale']
		self.cell_type = kwargs['cell_type']
		
		self.keep_prob = None
		self.back_propagate_all_layers = None
		self.enc_input = None
		self.dists_part3 = None
		self.dists_part5 = None
		self.dists_part35 = None
		
		self.correct_pred_part3 = None
		self.correct_pred_part5 = None
		self.accuracy_part3 = None
		self.accuracy_part5 = None
		self.train = None
		self.init_op = None
		self.enc_state_bw = None
		self.enc_state_fw = None
		self.enc_state_conc = None
		self.part3_final_op_softmax = None
		self.part5_final_op_softmax = None
		self.total_batch_loss = None
		
	def get_lstm_state(self, cell):
		if self.cell_type == 'GRU':
			return cell
		elif self.cell_type == 'LSTM':
			return cell.c
	
	def evaluate(self):
		
		model_start_time = dt.now()
		
		with tf.device("/cpu:0"):
			tf.reset_default_graph()
			
			# Training data placeholders
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
			self.back_propagate_all_layers = tf.placeholder(tf.bool, name="back_propagate_all_layers")
			self.enc_input = tf.placeholder(  # B x J x S
				tf.float32,
				name="enc_input",
				shape=(None, None, self.input_dimensions))
			self.dists_part3 = tf.placeholder(  # B x C3
				tf.int32,
				name="dists_part3",
				shape=(None,))
			self.dists_part5 = tf.placeholder(  # B x C5
				tf.int32,
				name="dists_part5",
				shape=(None,))
			part3_distribution = tf.one_hot(self.dists_part3, self.n_op_part3)
			part5_distribution = tf.one_hot(self.dists_part5, self.n_op_part5)
			# self.dists_part3 = tf.placeholder(  # B x C3
			# 	tf.float32,
			# 	name="dists_part3",
			# 	shape=(None, self.n_op_part3))
			# self.dists_part5 = tf.placeholder(  # B x C5
			# 	tf.float32,
			# 	name="dists_part5",
			# 	shape=(None, self.n_op_part5))
			if self.cell_type == 'GRU':
				cell_enc = tf.contrib.rnn.GRUCell(
					self.lstm_width_enc)
			else:
				cell_enc = tf.contrib.rnn.LSTMCell(
					self.lstm_width_enc,
					use_peepholes=False)
			
			cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=self.keep_prob)
			regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_scale)
			
			batch_size = array_ops.shape(self.enc_input)[0]
			
			# ################### Encoder ###################
			initial_state = cell_enc.zero_state(batch_size, tf.float32)  # B x L: 0 is starting state for RNN
			
			with tf.variable_scope("rnn_encoder"):
				d_rnn_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=cell_enc,
					cell_bw=cell_enc,
					dtype=tf.float32,
					initial_state_fw=initial_state,
					initial_state_bw=initial_state,
					inputs=self.enc_input)
			
			with tf.variable_scope("mlp"):
				# Biases
				b_part3_1 = tf.Variable(tf.random_normal([self.part3_hidden_1]), name="b_part3_1")  # D3_1
				b_part3_2 = tf.Variable(tf.random_normal([self.part3_hidden_2]), name="b_part3_2")  # D3_2
				b_part3_final = tf.Variable(tf.random_normal([self.n_op_part3]), name="b_part3_final")  # C3
				
				b_part5_1 = tf.Variable(tf.random_normal([self.part5_hidden_1]), name="b_part5_1")  # D5_1
				b_part5_2 = tf.Variable(tf.random_normal([self.part5_hidden_2]), name="b_part5_2")  # D5_2
				b_part5_final = tf.Variable(tf.random_normal([self.n_op_part5]), name="b_part5_final")  # C5
				
				# Weights
				with tf.variable_scope("Weights"):
					w_part3_1 = tf.get_variable("w_part3_1", [2 * self.lstm_width_enc, self.part3_hidden_1],
						regularizer=regularizer)  # L x D3_1
					w_part3_2 = tf.get_variable("w_part3_2", [self.part3_hidden_1, self.part3_hidden_2],
						regularizer=regularizer)  # D3_1 x D3_2
					w_part3_final = tf.get_variable("w_part3_final", [self.part3_hidden_2, self.n_op_part3],
						regularizer=regularizer)  # D3_2 x C3
					
					w_part5_1 = tf.get_variable("w_part5_1", [2 * self.lstm_width_enc, self.part5_hidden_1],
						regularizer=regularizer)  # L x D5_1
					w_part5_2 = tf.get_variable("w_part5_2", [self.part5_hidden_1, self.part5_hidden_2],
						regularizer=regularizer)  # D5_1 x D5_2
					w_part5_final = tf.get_variable("w_part5_final", [self.part5_hidden_2, self.n_op_part5],
						regularizer=regularizer)  # D5_2 x C5
				
				with tf.variable_scope("enc_op"):
					self.enc_state_fw = self.get_lstm_state(enc_states[0])
					self.enc_state_bw = self.get_lstm_state(enc_states[1])
					self.enc_state_conc = tf.concat([self.enc_state_fw, self.enc_state_bw], 1)
				
				with tf.variable_scope("final_op"):
					part3_layer_1 = tf.nn.relu(tf.matmul(self.enc_state_conc, w_part3_1) + b_part3_1)  # [B x 2L][2L x D3_1] = B x D3_1
					part3_layer_2 = tf.nn.relu(tf.matmul(part3_layer_1, w_part3_2) + b_part3_2)  # [B x D3_1][D3_1 x D3_2] = B x D3_2
					part3_dropout = tf.nn.dropout(part3_layer_2, self.keep_prob)
					part3_final_op = tf.matmul(part3_dropout, w_part3_final) + b_part3_final  # [B x D3_2][D3_2 x C3] = B x C3
					self.part3_final_op_softmax = tf.nn.softmax(part3_final_op)
			
					part5_layer_1 = tf.nn.relu(tf.matmul(self.enc_state_conc, w_part5_1) + b_part5_1)  # [B x 2L][2L x D5_1] = B x D5_1
					part5_layer_2 = tf.nn.relu(tf.matmul(part5_layer_1, w_part5_2) + b_part5_2)  # [B x D5_1][D5_1 x D5_2] = B x D5_2
					part5_dropout = tf.nn.dropout(part5_layer_2, self.keep_prob)
					part5_final_op = tf.matmul(part5_dropout, w_part5_final) + b_part5_final  # [B x D5_2][D5_2 x C5] = B x C5
					self.part5_final_op_softmax = tf.nn.softmax(part5_final_op)
			
			# ############## LOSS ###################
			with tf.variable_scope("loss"):
				with tf.variable_scope("loss_part3"):
					loss_part3 = tf.reduce_mean(
						tf.nn.softmax_cross_entropy_with_logits(labels=part3_distribution, logits=part3_final_op))
					reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
					reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
					loss_part3 += reg_term
					
				with tf.variable_scope("loss_part5"):
					loss_part5 = tf.reduce_mean(
						tf.nn.softmax_cross_entropy_with_logits(labels=part5_distribution, logits=part5_final_op))
					reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
					reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
					loss_part5 += reg_term
					
				with tf.variable_scope("total_batch_loss"):
					self.total_batch_loss = loss_part3 + loss_part5
		
			# ############## Accuracy ###################
			with tf.variable_scope("accuracy"):
				with tf.variable_scope("accuracy_part3"):
					self.correct_pred_part3 = tf.equal(
						tf.argmax(self.part3_final_op_softmax, 1), tf.argmax(part3_distribution, 1), name="equ3")
					self.accuracy_part3 = tf.reduce_mean(tf.cast(part5_distribution, tf.float32))
			
				with tf.variable_scope("accuracy_part5"):
					self.correct_pred_part5 = tf.equal(
						tf.argmax(self.part5_final_op_softmax, 1), tf.argmax(part5_distribution, 1), name="equ5")
					self.accuracy_part5 = tf.reduce_mean(tf.cast(self.correct_pred_part5, tf.float32))
			
			# ############## Train ###################
			self.train = tf.cond(
				self.back_propagate_all_layers,
				lambda: self.optimizer.minimize(self.total_batch_loss),
				lambda: self.optimizer.minimize(self.total_batch_loss, var_list=[
					b_part3_1, b_part3_2, b_part3_final, b_part5_1, b_part5_2, b_part5_final,
					w_part3_1, w_part3_2, w_part3_final, w_part5_1, w_part5_2, w_part5_final]))
			
			self.init_op = tf.global_variables_initializer()
			self.sess = tf.Session()
			self._saver = tf.train.Saver()
		
		logger.info("Model loaded in: " + str(dt.now() - model_start_time))
