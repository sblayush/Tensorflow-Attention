from pylab import *
import numpy as np


def plot_attention(attentions, ids, path):
	subplots_adjust(hspace=1.0)
	number_of_subplots = attentions.shape[0]

	for i, v in enumerate(range(number_of_subplots)):
		v = v+1
		ax1 = subplot(number_of_subplots, 1, v)
		ax1.plot(attentions[i])
		ax1.set_title(ids[i], loc="right")
	# plt.show()
	plt.savefig(path, dpi=100)
	plt.clf()
	plt.cla()
	plt.close()


if __name__ == '__main__':
	path = 'C:/Users/ab38686/Desktop/pyCharm/summarization/AspectBasedSentimentClassification/data/attention/1.png'
	y = np.random.rand(5, 20)
	ids = ["as"]*5
	print(ids)
	plot_attention(y, ids, path)
