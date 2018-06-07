import tensorflow as tf

v1 = tf.get_variable(
    name='v1', shape=[1], initializer=tf.constant_initializer(100))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable(
    name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    v1, v2 = tf.get_collection('loss')
    print(v1)
    print(v2)
    print(sess.run(tf.add_n(tf.get_collection('loss'))))
