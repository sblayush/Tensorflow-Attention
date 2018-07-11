import json
import numpy as np
from AspectBasedSentimentClassification.data_processing.maps import polarity_map, aspect_category_map
import nltk
import os
from autocorrect import spell
from AspectBasedSentimentClassification.data_processing.get_word_embeddings import get_glove_word_embeddings


class DataProcessing:
	
	def __init__(self, training=False, embedding_size=100):
		if training:
			self.restaurant_train_xml_path = "C:/Users/ab38686/Desktop/datasets/cnn_news/unprocessed/cnn/stories"
			# self.restaurant_train_xml_path = "C:/Users/ab38686/Desktop/datasets/cnn_news/processed/train/cnn/questions/training"
		else:
			self.restaurant_train_xml_path = "C:/Users/ab38686/Desktop/datasets/cnn_news/unprocessed/cnn/stories"
			# self.restaurant_train_xml_path = "C:/Users/ab38686/Desktop/datasets/cnn_news/processed/test/cnn/questions/training"
		self.n_polarity = len(polarity_map)
		self.n_aspect_category = len(aspect_category_map)
		self.n_vec = embedding_size
		self.glove_embeddings = get_glove_word_embeddings(self.n_vec)
		self.batch_size = 25
		self.norm_glove_embeddings, self.vocab, self.inv_vocab = self._generate()
	
	def _one_hot(self, pos, depth):
		arr = np.zeros([depth], dtype=int)
		arr[pos] = 1
		return arr
	
	def _get_word_vec(self, words):
		word_vecs = []
		for word in words:
			try:
				word_vec = self.glove_embeddings[word]
			except:
				try:
					word_vec = self.glove_embeddings[spell(word)]
					# print("{} = {}".format(word, spell(word)))
				except:
					# print("{} not found!".format(word))
					word_vec = np.zeros([self.n_vec])
			word_vecs.append(word_vec)
		return word_vecs
	
	def process_data(self, file):
		text = ""
		summaries = []
		flag = False
		for line in file:
			if line == "\n":
				continue
			if flag:
				if line[-1] == "\n":
					line = line[:-1]
				summaries.append(line)
				flag = False
			elif line[:-1] != "@highlight":
				text += line[:-1] + ' '
			if line[:-1] == "@highlight":
				flag = True
		summaries = '. '.join(summaries) + '.'
		return text, summaries
	
	def data_pre_processing(self):
		files = os.listdir(self.restaurant_train_xml_path)[:20000]
		batches = {}
		for file in files:
			with open(self.restaurant_train_xml_path + '/' + file, 'r', encoding='utf-8') as f:
				text, summaries = self.process_data(f)
			text_words = nltk.word_tokenize(text)[:400]
			summary_words = nltk.word_tokenize(summaries)[:100]
			if len(text_words) > 0 and len(summary_words) > 0:
				len_key = 'b'+str(len(text_words))+'_'+str(len(summary_words))
				if len_key not in batches:
					batches[len_key] = {
						'enc_inp': [],
						'dec_inp': []
					}
				enc_word_vecs_arr = self._get_word_vec(text_words)
				dec_word_vecs_arr = self._get_word_vec(summary_words)
				batches[len_key]['enc_inp'].append(enc_word_vecs_arr)
				batches[len_key]['dec_inp'].append(dec_word_vecs_arr)
				# batches[len_key]['vocab'].append([])
		print("{} batches created!".format(len(batches)))
		return batches
	
	def test_processing(self, text):
		batches = {}
		text_words = nltk.word_tokenize(text)[:400]
		len_key = 'b' + str(len(text_words))
		batches[len_key] = {
			'enc_inp': [self._get_word_vec(text_words)]
		}
		return batches
	
	def _generate(self):
		words = self.glove_embeddings.keys()
		vocab_size = len(words)
		vocab = {w: idx for idx, w in enumerate(words)}
		inv_vocab = {idx: w for idx, w in enumerate(words)}
		
		W = np.zeros((vocab_size, self.n_vec))
		for word, v in self.glove_embeddings.items():
			if word == '<unk>':
				continue
			W[vocab[word], :] = v
		
		# normalize each word vector to unit variance
		d = (np.sum(W ** 2, 1) ** 0.5)
		W_norm = (W.T / d).T
		return W_norm, vocab, inv_vocab
	
	def get_sent_form_vec(self, out_vecs):
		"""
		To generate words from output vectors using cosine similarity
		:param out_vecs: have dimentions [T x F] i.e., Time steps/(number of words in sentence) x feature size/vec_n
		:return: sentence
		"""
		# Top n words
		# sents = []
		# for word_vec in out_vecs:
		# 	dist = np.dot(self.norm_glove_embeddings, word_vec.T)
		#
		# 	# top_words = np.argsort(-dist)[:5]
		# 	top_words = [np.argmin(-dist)]
		# 	words = [self.inv_vocab[idx] for idx in top_words]
		# 	print(words)
		# 	sents.append(words)
		
		# Top words
		sent_arr = []
		for word_vec in out_vecs:
			dist = np.dot(self.norm_glove_embeddings, word_vec.T)
			top_idx = np.argmin(-dist)
			word = self.inv_vocab[top_idx]
			sent_arr.append(word)
		sent_text = ' '.join(sent_arr)
		return sent_text


def data_pre_processing(embedding_size=100):
	DPO = DataProcessing(embedding_size=embedding_size)
	return DPO.data_pre_processing()
	

if __name__ == '__main__':
	data_process = DataProcessing(training=True, embedding_size=100)
	# batchess = data_process.data_pre_processing()
	# print(batchess.keys())
	# for batch in batchess:
	# 	current_batch = batchess[batch]
	fly = data_process.glove_embeddings['fly']
	cat = data_process.glove_embeddings['run']
	dog = data_process.glove_embeddings['flee']
	ambiance = data_process.glove_embeddings['hitting']
	data_process.get_sent_form_vec(np.asarray([[ 1.70255378e-01,  1.79569483e-01,  1.54953212e-01,
         -4.84277681e-03,  6.59092665e-02, -1.96553804e-02,
          1.14779249e-01,  7.01989830e-02,  1.01339683e-01,
         -4.49148640e-02,  6.86050355e-02,  1.58829689e-02,
          2.72452049e-02, -3.09889391e-03, -9.00237486e-02,
          1.16841689e-01, -1.28012896e-01,  2.27394029e-02,
         -3.35882977e-02,  7.86038041e-02,  4.94873300e-02,
          7.80196264e-02,  2.78533045e-02,  1.75352506e-02,
         -1.17879480e-01,  3.19210142e-02, -2.69918144e-03,
          9.16600227e-02,  7.51702413e-02, -1.76085159e-01,
         -1.73041016e-01,  1.89821139e-01, -5.24990633e-02,
         -7.52218813e-02, -4.75697219e-02,  4.39130887e-03,
         -1.03482276e-01,  8.84247720e-02, -2.72922814e-02,
          9.63121876e-02, -7.00111762e-02, -9.58916545e-02,
         -8.76964331e-02, -1.53090298e-01,  1.58474222e-02,
         -1.19046152e-01, -4.92300279e-02, -5.59290722e-02,
          4.50627245e-02, -1.59518011e-02, -1.12877689e-01,
         -9.84176993e-04,  5.24447337e-02,  8.26118439e-02,
          1.33185446e-01, -2.00559452e-01,  5.97329661e-02,
          1.16518168e-02,  6.21878877e-02, -3.88042480e-02,
         -1.17391981e-02,  6.37048408e-02, -9.24066734e-03,
         -9.06904936e-02, -1.08685782e-02, -3.38004008e-02,
         -3.59790958e-03,  1.44879296e-02,  7.99380541e-02,
         -8.68658535e-03, -9.62685645e-02,  6.69183135e-02,
         -1.75958589e-01,  1.14665374e-01,  8.79336894e-02,
          1.40795127e-01, -4.49349880e-02, -7.63318613e-02,
          1.22778788e-02, -9.21067372e-02,  3.70527208e-02,
          1.08139999e-02,  1.05372883e-01,  2.40709819e-02,
         -2.79755816e-02, -1.28434990e-02,  2.37515345e-02,
         -6.39986917e-02,  1.08392745e-01,  1.62080958e-01,
          5.02836332e-02, -3.01980115e-02,  9.52557549e-02,
         -6.11635223e-02,  1.61131956e-02,  7.24681988e-02,
          1.38845399e-01, -4.96027879e-02,  1.32273823e-01,
         -1.89371631e-02],
        [ 3.02797318e-01,  4.54881042e-01,  3.32627594e-01,
         -7.24548995e-02,  2.36708865e-01, -1.20055534e-01,
          2.09921241e-01,  1.59449801e-01,  8.52103159e-02,
         -1.58676073e-01,  2.78862923e-01,  6.78881481e-02,
         -4.53205593e-02,  8.83530825e-02, -2.54038841e-01,
          2.48194754e-01, -3.61456871e-01, -2.46005431e-02,
         -2.91117191e-01,  9.29111615e-02,  1.21666968e-01,
          2.08660334e-01,  2.94688232e-02,  1.59223564e-04,
         -8.95922184e-02,  1.97193474e-01, -5.67382686e-02,
          1.88426659e-01,  2.14879990e-01, -3.88004988e-01,
         -4.48970795e-01,  6.08603537e-01, -4.65802774e-02,
         -5.43828271e-02, -1.62809901e-02,  8.08677524e-02,
         -1.62480831e-01,  2.93383420e-01, -4.07722816e-02,
          1.91423163e-01, -1.53178439e-01, -1.60689548e-01,
         -2.45820329e-01, -4.42930818e-01, -1.32017002e-01,
         -9.77991074e-02, -3.51489335e-03, -1.25174329e-01,
          1.94149628e-01, -6.48002997e-02, -3.74544322e-01,
          4.12835032e-02,  2.03542292e-01,  1.72399491e-01,
          2.90367335e-01, -6.76559389e-01,  2.16901898e-01,
          1.05194263e-01, -6.37302846e-02, -3.31617445e-02,
          9.79291126e-02,  9.06376541e-02,  1.07613783e-02,
          1.91132426e-02, -6.69448152e-02, -1.14832424e-01,
          5.85830063e-02,  2.63023138e-01,  2.37956002e-01,
         -6.83058128e-02, -1.84132710e-01,  1.18018605e-01,
         -3.33184838e-01,  3.49068105e-01,  1.28145754e-01,
          3.38543802e-01, -1.67016447e-01, -1.39910161e-01,
         -4.06533740e-02, -2.88046896e-01,  1.64416686e-01,
          1.57856330e-01,  1.06524438e-01,  9.01558623e-02,
         -8.25333521e-02, -6.88139163e-03, -2.55841762e-04,
         -2.31824443e-01,  3.76193881e-01,  5.06444871e-01,
          9.99531373e-02, -1.47729337e-01,  1.61534756e-01,
         -1.97574124e-03, -1.36586949e-01,  2.42681608e-01,
          2.39140257e-01, -5.91443442e-02,  2.06273928e-01,
         -1.31032374e-02], dog, ambiance, cat, fly]))
