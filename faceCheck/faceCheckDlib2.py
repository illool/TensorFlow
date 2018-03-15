#coding=utf-8
# 2017-11-27
# by TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/

import dlib
from skimage import io

# 使用特征提取器frontal_face_detector
detector = dlib.get_frontal_face_detector()

# dlib的68点模型
path_pre = "E:\\face_recognition_models-0.3.0\\face_recognition_models\\models"
predictor = dlib.shape_predictor(path_pre+"\\shape_predictor_68_face_landmarks.dat")

# 图片所在路径
img = io.imread("3.jpg")

# 生成dlib的图像窗口
win = dlib.image_window()
win.clear_overlay()
win.set_image(img)

# 特征提取器的实例化
dets = detector(img, 1)
print("人脸数：", len(dets))

shape2 = predictor(img, dets[0])
win.add_overlay(shape2)

for k, d in enumerate(dets):
        print("第", k, "个人脸d的坐标：",
              "left:", d.left(),
              "right:", d.right(),
              "top:", d.top(),
              "bottom:", d.bottom())

        # 利用预测器预测
        shape = predictor(img, d)

        # 绘制面部轮廓
#        win.add_overlay(shape)

# 绘制矩阵轮廓
win.add_overlay(dets)
# 保持图像
dlib.hit_enter_to_continue()
