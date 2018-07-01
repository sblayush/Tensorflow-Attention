import tensorflow as tf


print("imported tf")
b = 5
d1 = 2
t = 3
d2 = 4


A = tf.Variable(tf.random_normal(shape=(b, t, d1)))
B = tf.Variable(tf.random_normal(shape=(b, d2)))
batch_size = tf.placeholder(dtype=tf.int64, shape=[])
print(batch_size)
C = tf.reshape(A, [-1, d1])
print(batch_size.get_shape())
D = tf.reshape(C, tf.stack([batch_size, -1, d1]))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

l = sess.run(D, feed_dict={batch_size: b})
print(l.shape)
