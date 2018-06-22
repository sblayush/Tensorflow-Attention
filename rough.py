import tensorflow as tf


print("imported tf")
a = 5
m = 2
n = 3
k = 4


A = tf.Variable(tf.random_normal(shape=(a, n, m)))
B = tf.Variable(tf.random_normal(shape=(m, k)))
C = tf.tile(B, [a, 1])
d = tf.matmul(A, C)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with sess.as_default():
	l = d.eval()
	print(l)
	print(l.shape)