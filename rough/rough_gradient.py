import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function
import numpy as np

a = tf.add(1, 2, name="Add_these_numbers")
b = tf.multiply(a, 3, name='mult')

mult = tf.get_default_graph().get_operation_by_name('mult')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>

tf.stop_gradient(a, name='stop')
stop = tf.get_default_graph().get_operation_by_name('stop')
print(get_gradient_function(stop))  # None

c = tf.squeeze(a, name="c")
mult = tf.get_default_graph().get_operation_by_name('c')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>

indices = np.asarray([[0, 0], [1, 1]])
params = np.asarray([['a', 'b'], ['c', 'd']])

ga = tf.gather_nd(
    params,
    indices,
    name="ga"
)

mult = tf.get_default_graph().get_operation_by_name('ga')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>
am = tf.argmax(a, name="am")
mult = tf.get_default_graph().get_operation_by_name('am')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>

indice = np.asarray([[0.1, 0.3], [1.4, 1.32]])
sm = tf.nn.softmax(indice, name="sm")
mult = tf.get_default_graph().get_operation_by_name('sm')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>


ra = tf.range(2, name="ra")
mult = tf.get_default_graph().get_operation_by_name('ra')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>

til = tf.tile(params, [1, 2], name="til")
mult = tf.get_default_graph().get_operation_by_name('til')
print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>

