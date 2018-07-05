import numpy as np
import tensorflow as tf
from AspectBasedSentimentClassification.SentimentClassificationModel import SentimentClassificationModel
from AspectBasedSentimentClassification.data_processing.data_preprocessing import DataProcessing
from AspectBasedSentimentClassification.data_processing.maps import reverse_polarity_map, reverse_aspect_category_map
from visualization.plot_attention import plot_attention

_EPOCHS = 150
_BATCH_SIZE = 10
_ENC_N = 56
_DEC_N = 56
_DEPTH = 100
_ASPECT_N = 10
_LEARNING_RATE = 0.001
summary_path = 'C:/Users/ab38686/Desktop/pyCharm/summarization/summaries/sentiment_classification'
viz_path = 'C:/Users/ab38686/Desktop/pyCharm/summarization/AspectBasedSentimentClassification/data/attention/{}.png'


if __name__ == '__main__':
	data_process = DataProcessing(training=True, embedding_size=_DEPTH)
	batches_train = data_process.data_pre_processing()
	my_model = SentimentClassificationModel(
		n_inp=_DEPTH,
		n_enc_h=_ENC_N,
		n_aspect_class=data_process.n_aspect_category,
		n_polarity=data_process.n_polarity,
		n_aspect_embed=_ASPECT_N,
		learning_rate=_LEARNING_RATE
	)
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
	
	cnt = 0
	for epoch in range(_EPOCHS):
		total_loss = 0
		pred_class = []
		for batch in batches_train:
			current_batch = batches_train[batch]
			f_dict = {
				my_model.X: current_batch['inp'],
				my_model.AspectClass: current_batch['aspects_arr'],
				my_model.Y: current_batch['aspects_polarity_arr']
			}
			pred_class, lo, _, summ, alpa = sess.run(
				[my_model.pred_class_softmax, my_model.loss, my_model.train, my_model.merge, my_model.alphas],
				feed_dict=f_dict
			)
			total_loss += lo
		print(total_loss)
		if epoch % 10 == 0:
			print(np.argmax(pred_class, axis=1))
			print(np.argmax(current_batch['aspects_polarity_arr'], axis=1))
			print((np.argmax(alpa, axis=2)).flatten())
			train_writer.add_summary(summ, epoch)
			print(current_batch['ids'])
			print([reverse_polarity_map[np.argmax(pol)] for pol in pred_class])
			plot_attention(alpa[:5, 0, :], current_batch['ids'][:5], viz_path.format(epoch))
	
	data_process_test = DataProcessing(training=False, embedding_size=_DEPTH)
	batches_test = data_process_test.data_pre_processing()
	
	total_loss = 0
	for batch in batches_test:
		current_batch = batches_test[batch]
		f_dict = {
			my_model.X: current_batch['inp'],
			my_model.AspectClass: current_batch['aspects_arr'],
			my_model.Y: current_batch['aspects_polarity_arr']
		}
		pred_class, lo, alpa = sess.run(
			[my_model.pred_class_softmax, my_model.loss, my_model.alphas], feed_dict=f_dict
		)
		total_loss += lo
	print(total_loss)
	print(np.argmax(pred_class, axis=1))
	print(np.argmax(current_batch['aspects_polarity_arr'], axis=1))
	print((np.argmax(alpa, axis=2)).flatten())
	print(current_batch['ids'])
	print([reverse_polarity_map[np.argmax(pol)] for pol in pred_class])
	plot_attention(alpa[:5, 0, :], current_batch['ids'][:5], viz_path.format('00'))