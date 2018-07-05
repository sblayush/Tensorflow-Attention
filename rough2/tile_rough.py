import tensorflow as tf
A = tf.random_normal([2, 10, 5])
B = tf.random_normal([2, 10, 5])
C = tf.random_normal([2, 10])
# C = tf.greater_equal(C, tf.zeros_like(C))

C = tf.tile(C, [1, tf.shape(A)[1], 1])

with tf.Session() as sess:
	print(sess.run(C))
	print(sess.run(C).shape)
