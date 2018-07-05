import json
import xmltodict
import numpy as np
from AspectBasedSentimentClassification.data_processing.maps import polarity_map, aspect_category_map
import nltk
from autocorrect import spell
from AspectBasedSentimentClassification.data_processing.get_word_embeddings import get_glove_word_embeddings


class DataProcessing:
	
	def __init__(self, training=False, embedding_size=100):
		if training:
			self.restaurant_train_xml_path = "C:/Users/ab38686/Desktop/restaurant_train/Restaurants_Train.xml"
		else:
			self.restaurant_train_xml_path = "C:/Users/ab38686/Desktop/restaurant_train/Restaurants_Test.xml"
		self.n_polarity = len(polarity_map)
		self.n_aspect_category = len(aspect_category_map)
		self.n_vec = embedding_size
		self.glove_embeddings = get_glove_word_embeddings(self.n_vec)
		self.batch_size = 25
	
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
	
	def data_pre_processing(self):
		with open(self.restaurant_train_xml_path, 'r') as f:
			xmlString = f.read()
		restaurant_json = json.loads(json.dumps(xmltodict.parse(xmlString)))
		# print(json.dumps(restaurant_json['sentences']['sentence'], indent=2))
		batches = {}
		for review in restaurant_json['sentences']['sentence']:
			aspect_category = review['aspectCategories']['aspectCategory']
			text = review['text'].lower()
			text_words = nltk.word_tokenize(text)
			len_key = 'b'+str(len(text_words))
			if len_key not in batches:
				batches[len_key] = {
					'ids': [],
					'inp': [],
					'aspects_polarity_arr': [],
					'aspects_arr': []
				}
			word_vecs_arr = self._get_word_vec(text_words)
			if type(aspect_category) is list:
				for aspect in aspect_category:
					batches[len_key]['ids'].append(review['@id'])
					batches[len_key]['inp'].append(word_vecs_arr)
					batches[len_key]['aspects_polarity_arr'].append(
						self._one_hot(polarity_map[aspect['@polarity']], self.n_polarity))
					batches[len_key]['aspects_arr'].append(
						self._one_hot(aspect_category_map[aspect['@category']], self.n_aspect_category))
			else:
				batches[len_key]['ids'].append(review['@id'])
				batches[len_key]['inp'].append(word_vecs_arr)
				batches[len_key]['aspects_polarity_arr'].append(
					self._one_hot(polarity_map[aspect_category['@polarity']], self.n_polarity))
				batches[len_key]['aspects_arr'].append(
					self._one_hot(aspect_category_map[aspect_category['@category']], self.n_aspect_category))
		print("{} batches created!".format(len(batches)))
		return batches
	
	
def data_pre_processing(embedding_size=100):
	DPO = DataProcessing(embedding_size)
	return DPO.data_pre_processing()
	

if __name__ == '__main__':
	data_process = DataProcessing(training=False, embedding_size=100)
	batches = data_process.data_pre_processing()
	for batch in batches:
		current_batch = batches[batch]
	
