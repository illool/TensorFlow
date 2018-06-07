# coding: utf-8
import tensorflow as tf
# 生成一个先入先出队列和一个QueueRunner,生成文件名队列
filenames = ['irs.csv', 'irs.csv', 'irs.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species = tf.decode_csv(value, record_defaults=[
    [0.], [0.], [0.], [0.], ['']])
#example_batch, label_batch = tf.train.shuffle_batch([example,label], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)
# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(
        coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
    for i in range(10):
        print(SepalLengthCm.eval(), SepalWidthCm.eval(),
              PetalLengthCm.eval(), PetalWidthCm.eval(), Species.eval())
    coord.request_stop()
    coord.join(threads)
