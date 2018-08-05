import numpy as np
from Summarization.config_paths import word_embeddings_path


def get_glove_word_embeddings(embedding_size):
	with open(word_embeddings_path + '/glove.6B.' + str(embedding_size) + 'd.txt', 'r', encoding="utf8") as f:
		model = {}
		for line in f:
			split_line = line.split()
			word = split_line[0]
			embedding = np.asarray([float(val) for val in split_line[1:]])
			model[word] = embedding
	print("Done. " + str(len(model)) + " words loaded!")
	return model
