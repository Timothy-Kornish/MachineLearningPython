import tensorflow as tf
import numpy as np


hello = tf.constant('Hello World')
x = tf.constant(100)
print("\n-------------------------------------------------------------------\n")
print(type(hello))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Creating a Tensor Flow Session
#-------------------------------------------------------------------------------

sess = tf.Session()
print("\n-------------------------------------------------------------------\n")
print(sess.run(hello))
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Creating a Tensor Flow Operations
#-------------------------------------------------------------------------------

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print("\n-------------------------------------------------------------------\n")
    print("Operations with constants")
    print("Addition: ", sess.run(x + y))
    print("Subtraction: ", sess.run(x - y))
    print("Multiplication: ", sess.run(x * y))
    print("Division: ", sess.run(x / y))
    print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Creating a Tensor Flow Placeholders
#-------------------------------------------------------------------------------

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

print("\n-------------------------------------------------------------------\n")
print('X:', x)
print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Performing Tensor Flow Operations With Placeholders
#-------------------------------------------------------------------------------

add = tf.add(x, y)
sub = tf.subtract(x, y)
mul = tf.multiply(x, y)

with tf.Session() as sess:
    print("\n-------------------------------------------------------------------\n")
    print("Operations with placeholders")
    print("Addition: ", sess.run(add, feed_dict = {x: 20, y: 30}))
    print("Subtraction: ", sess.run(sub,  feed_dict = {x: 20, y: 30}))
    print("Multiplication: ", sess.run(mul,  feed_dict = {x: 20, y: 30}))
    print("\n-------------------------------------------------------------------\n")

#-------------------------------------------------------------------------------
#           Converting Numpy arrays into Tensor Flow objects
#-------------------------------------------------------------------------------

a = np.array([[5.0, 5.0]])
b = np.array([[2.0], [2.0]])

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print("\n-------------------------------------------------------------------\n")
    print("matrix multiplication: ", result)
    print("\n-------------------------------------------------------------------\n")
