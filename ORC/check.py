#coding=utf-8
import tensorflow as tf
from OCRTrain import *
from imagegen import *

def crack_captcha(captcha_image):
    output = crack_captcha_cnn()
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
 
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text
 
text, image = gen_captcha_text_and_image()
image = convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text)) 
