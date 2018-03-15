# coding: utf-8
import cv2
import sys
import dlib
from skimage import io
import time

# 使用特征提取器frontal_face_detector
# 利用dlib的特征提取器，进行人脸 矩形框 的提取
detector = dlib.get_frontal_face_detector()

# dlib的68点模型
path_pre = "E:\\face_recognition_models-0.3.0\\face_recognition_models\\models"
predictor = dlib.shape_predictor(path_pre+"\\shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor(path_pre+"\\shape_predictor_5_face_landmarks.dat")
cap = cv2.VideoCapture(0)
win = dlib.image_window()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # 生成dlib的图像窗口
        #win = dlib.image_window()
        win.clear_overlay()
        win.set_image(frame)

        # 特征提取器的实例化
        # 利用dlib的特征提取器，进行人脸 矩形框 的提取
        dets = detector(frame, 1)
        print("人脸数：", len(dets))
        if len(dets) <= 0:
            continue
        for i in range(len(dets)):
            #shape2 = predictor(frame, dets[0])
            #win.add_overlay(shape2)
        
            for k, d in enumerate(dets):
                print("第", k, "个人脸d的坐标：",
                      "left:", d.left(),
                      "right:", d.right(),
                      "top:", d.top(),
                      "bottom:", d.bottom())
    
                # 利用预测器预测
                shape = predictor(frame, d)
    
                # 绘制面部轮廓
                win.add_overlay(shape)
                # 绘制矩阵轮廓
                win.add_overlay(dets)
                # 保持图像
                #dlib.hit_enter_to_continue()
                time.sleep(1)   
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
