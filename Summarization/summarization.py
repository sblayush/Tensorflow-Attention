import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)).replace('\\', '/'))
sys.path.append(parent_dir)
import numpy as np
import tensorflow as tf
from Summarization.SummarizationModel import SummarizationModel
from Summarization.data_processing.data_preprocessing import DataProcessing
from Summarization.config_paths import summaries_path, models_path
from datetime import datetime as dt
from util.common_functions import print_progress_bar

_EPOCHS = 300
_EPOCH_STEP = 1
_BATCH_SIZE = 100
_ENC_N = 56
_DEC_N = 56
_DEPTH = 50
_ASPECT_N = 10
_LEARNING_RATE = 0.001
_TRAIN = False


if __name__ == '__main__':
	my_model = SummarizationModel(
		n_inp=_DEPTH,
		n_enc_h=_ENC_N,
		n_dec_h=_DEC_N,
		learning_rate=_LEARNING_RATE
	)
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()
	
	if _TRAIN:
		train_writer = tf.summary.FileWriter(summaries_path + '/train', sess.graph)
		data_process = DataProcessing(training=_TRAIN, embedding_size=_DEPTH, batch_size=_BATCH_SIZE)
		# batches_train = data_process.data_pre_processing()

		start = dt.now()
		end = None
		cnt = 0
		for epoch in range(_EPOCHS):
			total_loss = 0
			pred_class = []
			if cnt > data_process.batches_len:
				data_process.batches_len = cnt
			cnt = 0
			for current_batch in data_process.get_next_batch():
				# current_batch = batches_train[batch]
				oup = np.asarray(current_batch['dec_inp'])
				z = np.ones((oup.shape[0], 1, _DEPTH)) * -1
				oup = np.flip(np.append(np.flip(oup, axis=1), z, axis=1), axis=1)
				
				f_dict = {
					my_model.X: current_batch['enc_inp'],
					my_model.Y: oup,
					my_model.is_running: not _TRAIN
				}
				lo, _, summ = sess.run(
					[my_model.loss, my_model.train, my_model.merge],
					feed_dict=f_dict
				)
				total_loss += lo
				print_progress_bar(cnt, data_process.batches_len)
				cnt += 1
				end = dt.now()
			if epoch % _EPOCH_STEP == 0:
				saver.save(sess, models_path + '/my_model2/train', global_step=epoch)
			print('*' * 50)
			print("EPOCH: {}".format(epoch))
			print("TIME: " + str(end - start))
			print("LOSS:", total_loss)
			print('*'*50)
			start = end
	
	else:
		saver.restore(sess, tf.train.latest_checkpoint(models_path + '/my_model2/'))
		data_process = DataProcessing(training=_TRAIN, embedding_size=_DEPTH, batch_size=_BATCH_SIZE)
		while True:
			sent = input("Enter sentence:")
			# sent = "Argentina lost to Germany in a well fought match 3-0"
			batches = data_process.test_processing(sent)
			for batch_key in batches:
				current_batch = batches[batch_key]
				f_dict = {
					my_model.X: current_batch['enc_inp'],
					my_model.Y: current_batch['enc_inp'],
					my_model.is_running: True
				}
				saver.restore(sess, tf.train.latest_checkpoint(models_path + '/my_model2'))
				con_ar = sess.run([my_model.context_arr], feed_dict=f_dict)
				out_sent = data_process.get_sent_form_vec(con_ar[0][0])
				print(out_sent)
