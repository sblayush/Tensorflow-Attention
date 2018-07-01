import tensorflow as tf
import numpy as np


def index_matrix_to_pairs(index_matrix):
	# [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
	#                        [[0, 2], [1, 3], [2, 1]]]
	replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
	rank = len(index_matrix.get_shape())
	if rank == 2:
		replicated_first_indices = tf.tile(
			tf.expand_dims(replicated_first_indices, dim=1),
			[1, tf.shape(index_matrix)[1]])
	return tf.stack([replicated_first_indices, index_matrix], axis=rank)


if __name__ == '__main__':

	x = np.random.rand(10, 1, 10)
	y = np.random.rand(10, 10, 1)

	X = tf.constant(x)
	Y = tf.constant(y)
	S = tf.nn.softmax(X, axis=2)
	S = tf.argmax(S, axis=2, output_type=tf.int32)
	B = index_matrix_to_pairs(S)
	outputs = tf.gather_nd(params=Y, indices=index_matrix_to_pairs(S))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	b, s = sess.run([B, S])
	print(x)
	print()
	print(s)
	print()
	print(b)

	o = sess.run(outputs)
	print()
	print(o)
