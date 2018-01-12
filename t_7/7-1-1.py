#coding=utf-8
#将mnist输入数据保存为TFRecord格式
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#生成整数型的属性
def _int64_feature(value):
    #__init__(**kwargs)
    #构造一个Feature对象,一般使用的时候,传入 tf.train.Int64List, tf.train.BytesList, tf.train.FloatList对象.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("F:\\PythonPro\\testTensorFlow\\src\\MNIST_data",dtype=tf.uint8,one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
#训练数据的图像分辨率，这可以作为一个属性保存在TFRecord里
#55000*784
pixels = images.shape[1]
print(images.shape)
#55000
num_examples = mnist.train.num_examples
print(num_examples)

#输出TFRecord文件的地址
filename = './output.tfrecords'
#创建一个writer来写TFRecord文件
#__init__(path,options=None)直接调用初始化函数
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):#0:55000
    #将图像矩阵转为一个字符串
    image_raw = images[index].tostring()
    #将一个样例转化为一个EXample Protocal Buffer，并将所有的信息写入这个数据结构
    #__init__(**kwargs)
    #这个函数是初始化函数,会生成一个Example对象,一般我们使用的时候,是传入一个tf.train.Features对象进去.
    example = tf.train.Example(features=tf.train.Features(feature={#__init__(**kwargs)；初始化Features对象,一般我们是传入一个字典,字典的键是一个字符串,表示名字,字典的值是一个tf.train.Feature对象
        'pixels':_int64_feature(pixels),#有多少个像素点
        'label':_int64_feature(np.argmax(labels[index])),#标签是什么
        'image_raw':_bytes_feature(image_raw)}))#实际数据
    #将一个Example写入TFRecord文件
    writer.write(example.SerializeToString())
writer.close()
