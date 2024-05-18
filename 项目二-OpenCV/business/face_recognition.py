import base64

import cv2  # 导入OpenCV库
import os  # 导入os模块，用于文件路径操作

import numpy as np

#创建LBPH人脸识别器对象
recognizer = cv2.face.LBPHFaceRecognizer_create()  #是不是要开了摄像头才能用
# 读取已训练的人脸识别模型
recognizer.read(r'D:/paper/xfei/face_opencv/face_opencv/flaskProject/trained_model.yml')

# 存储人名列表
names = []


def name():
    # 设置图片路径
    path = 'D:/paper/xfei/face_id/test/face/'
    # 遍历路径下的图片文件
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower() != 'desktop.ini']
    # 提取每个图片文件的人名并添加到names列表中
    names = []
    for imagePath in imagePaths:
        name = str(os.path.split(imagePath)[1].split('.', 3)[
                       1])  # Assuming the name is in the third part after splitting by '.'
        print('name:', name)
        names.append(name)
    return names

# 调用name函数获取人名列表
names = name()


# 定义人脸检测函数
def face_detect_demo(img):
    # 解码base64
    # img = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载人脸检测器分类器
    face_detector = cv2.CascadeClassifier('D:/paper/xfei/face_opencv/face_opencv/flaskProject/haarcascade_frontalface_alt2.xml')
    # 检测图像中的人脸
    face = face_detector.detectMultiScale(gray, 1.2, 5)

    for x, y, w, h in face:
        # 在图像中标注出人脸位置
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

        # 使用人脸识别器识别人脸并显示对应的姓名
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print("ids:", ids, " cnofidence", confidence)

        if confidence < 80:
            cv2.putText(img, str(names[ids]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    return img











