import numpy as np
import tensorflow as tf
from Summarization.SummarizationModel import SummarizationModel
from Summarization.data_processing.data_preprocessing import DataProcessing
from visualization.plot_attention import plot_attention

_EPOCHS = 300
_EPOCH_STEP = 1
_BATCH_SIZE = 10
_ENC_N = 56
_DEC_N = 56
_DEPTH = 100
_ASPECT_N = 10
_LEARNING_RATE = 0.001
_TRAIN = True
summary_path = 'C:/Users/ab38686/Desktop/pyCharm/summarization/summaries/summarization'
viz_path = 'C:/Users/ab38686/Desktop/pyCharm/summarization/summarization/data/summarization/{}.png'
model_path = 'C:/Users/ab38686/Desktop/pyCharm/summarization/saved_models/summarization'


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
		train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
		data_process = DataProcessing(training=_TRAIN, embedding_size=_DEPTH)
		batches_train = data_process.data_pre_processing()
		cnt = 0
		for epoch in range(_EPOCHS):
			total_loss = 0
			pred_class = []
			for batch in batches_train:
				current_batch = batches_train[batch]
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
			print(total_loss)
			if epoch % _EPOCH_STEP == 0:
				saver.save(sess, model_path + '/my_model', global_step=epoch)
	
	else:
		data_process = DataProcessing(training=_TRAIN, embedding_size=_DEPTH)
		sent = "Argentina lost to Germany in a well fought match 3-0"
		batches_test = data_process.test_processing(sent)
		for batch in batches_test:
			# print(batch)
			current_batch = batches_test[batch]
			f_dict = {
				my_model.X: current_batch['enc_inp'],
				my_model.Y: current_batch['enc_inp'],
				my_model.is_running: True
			}
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
			con_ar = sess.run([my_model.context_arr], feed_dict=f_dict)
			out_sent = data_process.get_sent_form_vec(con_ar[0][0])
			print(out_sent)
