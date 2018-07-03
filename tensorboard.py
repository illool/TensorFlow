import tensorflow as tf

with tf.variable_scope('input1'):
    input1 = tf.constant([1.0,2.0,3.0],name='input1')

with tf.variable_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]),name='input2')

add = tf.add_n([input1,input2],name='addOP')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter(".//TensorBoard//test",sess.graph)
    print(sess.run(add))
    print("on cmd line input => tensorboard --logdir=/xxxx/xxx/xxx/xxx....")
writer.close()
