#coding=utf-8
#tf里队列的使用

import tensorflow as tf
#声明一个先进先出的队列，指明有两个元素，类型是Int32
q = tf.FIFOQueue(capacity=2,dtypes=tf.int32)
#声明一个随机队列，参数：容量，出对列后的最小值，类型
#q = tf.RandomShuffleQueue(2,1,'int32')
#初始化队列
#里面放两个数：10，0
#通过enqueue_many()函数初始化队列中的元素
#定义初始化操作
#注意，如果一次性入列超过Queue Size的数据，enqueue操作会卡住，直到有数据（被其他线程）从队列取出。对一个已经取空的队列使用dequeue操作也会卡住，直到有新的数据（从其他线程）写入。
init = q.enqueue_many(([10,0],))
#通过dequeue()函数将队列中的第一个元素送出队列，并将这个元素的值存在变量x中，出队列
x=q.dequeue()
y=x+1
#将更新后的元素值重新加入队列中 ,入队列
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()#执行初始化操作
    for _ in range(5):
        #运行q_inc将执行数据出对列，加一，入队列的过程
        v,_= sess.run([x,q_inc])
        print(v)
        
    quelen = sess.run(q.size())  
    for i in range(quelen):  
        print(sess.run(q.dequeue()))  
