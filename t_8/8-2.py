#coding=utf-8
#简单LSTM 结构的RNN 的前向传播过程实现
import tensorflow as tf
from numpy.random import RandomState

lstm_hidden_size=1
batch_size=21
num_steps=22

#X=tf.constant([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]])
#X=tf.constant([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.])
#rdm = RandomState(1)
#X = rdm.rand(num_steps,2)
X = tf.Variable(tf.random_normal([2,num_steps-1,1],stddev=1,seed=1))
print(X.shape)
#定义一个LSTM结构。在TensorFlow中通过一句简单的命令就可以实现一个完整LSTM结构。
# LSTM中使用的变量也会在该函数中自动被声明。
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
# 将LSTM中的状态初始化为全0数组。和其他神经网络类似，在优化循环神经网络时，每次也
# 会使用一个batch的训练样本。以下代码中，batch_size给出了一个batch的大小。
# BasicLSTMCell类提供了zero_state函数来生成全领的初始状态。
state = lstm.zero_state(batch_size, tf.float32)
# 定义损失函数。
loss = 0.0
# 在8.1节中介绍过，虽然理论上循环神经网络可以处理任意长度的序列，但是在训练时为了
# 避免梯度消散的问题，会规定一个最大的序列长度。在以下代码中，用num_steps
# 来表示这个长度。
with tf.Session() as sess:
    for i in range(num_steps):
# 在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量。   
        if i > 0: tf.get_variable_scope().reuse_variables()
    
     # 每一步处理时间序列中的一个时刻。将当前输入（current_input）和前一时刻状态
     # （state）传入定义的LSTM结构可以得到当前LSTM结构的输出lstm_output和更新后
     # 的状态state。
        lstm_output, state = lstm(X[i], state)
   # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出。
        final_output = fully_connected(lstm_output)
   # 计算当前时刻输出的损失。
        loss += calc_loss(final_output, expected_output)
