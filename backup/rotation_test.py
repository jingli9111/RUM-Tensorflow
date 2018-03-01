import tensorflow as tf 
from RUM import rotate 
from RUM import rotation_operator 

sess = tf.Session()

d = tf.constant([[1.21,1.23,3.2,0.0,1.7,0.0]],shape=[1,6])
a = tf.constant([[4.2,5.22,7.0,2.0,3.3,4.0]],shape=[1,6])
b = tf.constant([[0.7,10.0,2.3,6.5,0.0,0.5]],shape=[1,6])
c = rotate(a, b, d)
e = rotation_operator(a, b)

print(sess.run(d))
print(sess.run(e))
print(sess.run(tf.matmul(e,tf.reshape(d,[1,6,1]))))
print(sess.run(c))
