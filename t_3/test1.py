#coding=utf-8
import tensorflow as tf
import numpy as np
'''
g1 = tf.Graph()
with g1.as_default():
	v=tf.get_variable("v",initializer=tf.zeros(shape=[1]))
g2 = tf.Graph()
with g2.as_default():
	v=tf.get_variable("v",initializer=tf.ones_initializer(shape=[1]))
with tf.Session(graph=g1) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))
with tf.Session(graph=g2) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("",reuse=True):
		print(sess.run(tf.get_variable("v")))
'''

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())

g=tf.Graph()
print("g:",g)
with g.as_default():
	d=tf.constant(value=2)
	print(d.graph)
	#print(g)

g2=tf.Graph()
print("g2:",g2)
g2.as_default()
e=tf.constant(value=15)
print(e.graph)

g1 = tf.Graph()
with g1.as_default():
	c1 = tf.constant([1.0])
with tf.Graph().as_default() as g2:
	c2 = tf.constant([2.0])
with tf.Session(graph=g1) as sess1:
	print("如果将上面例子的sess1.run(c1)和sess2.run(c2)中的c1和c2交换一下位置，运行会报错。因为sess1加载的g1中没有c2这个Tensor，同样地，sess2加载的g2中也没有c1这个Tensor。")
	print(sess1.run(c1))
with tf.Session(graph=g2) as sess2:
	print(sess2.run(c2))
	
