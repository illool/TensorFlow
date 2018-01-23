# encoding=utf-8
import pickle
import glob
import common
import utils
import random

__author__ = ""
a = ['a','b','c','d']
print (a.index('d'))

import cv2
import numpy as np
img = cv2.imread("1.jpg")
cv2.imshow("lena",img)
cv2.waitKey(10000)

import os, sys

# 打开文件
path = "models"
dirs = os.listdir( path )

# 输出所有文件和文件夹
for file in dirs:
   print(file)
   
list1 = ["这", "是", "一个", "测试"]
for index, item in enumerate(list1):
    print(index, item)
for index, item in enumerate(list1, 1):
    print(index, item)
    
fname_index1 = "{:08d}".format(1)
fname_index2 = "{:016d}".format(1)
print(fname_index1)
print(fname_index2)
#获得目录下的所有.png的文件
print(glob.glob(r"F:\PythonPro\Tensorflow-master\src\tensorflow_lstm_ctc_ocr-master\test\*.png"))

def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color

text_color, plate_color = pick_colors()
print(text_color, plate_color)

#for batch in xrange(common.BATCHES):
#    train_inputs, train_targets, train_seq_len = utils.get_data_set('train', batch*common.BATCH_SIZE, (batch + 1) * common.BATCH_SIZE)
#    print batch, train_inputs.shape

