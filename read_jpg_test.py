# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
        res = list(map(lambda x: file_dir + '\\' + x, files))
        print(res)  # 当前路径下所有非目录子文件
    return res


path = file_name(r'F:\SUN397\a\abbey')
# 利用tf.train.string_input_producer函数生成一个读取队列
file_queue = tf.train.string_input_producer(path)  # 创建输入队列

image_reader = tf.WholeFileReader()
_, image = image_reader.read(file_queue)
image = tf.image.decode_jpeg(image)


with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 协同启动的线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

    # 主线程
    for i in range(0, len(path)):
        sess.run(image)
        coord.request_stop()  # 停止所有的线程
        coord.join(threads)
        image_uint8 = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        plt.ion()
        plt.imshow(image_uint8.eval())
        # plt.show()
        plt.pause(0.01)
        plt.close()

    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    # join操作经常用在线程当中,其作用是等待某线程结束
    coord.join(threads)
