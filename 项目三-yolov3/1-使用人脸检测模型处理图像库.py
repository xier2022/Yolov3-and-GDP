# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from yolo import YOLO
from PIL import Image

# 这段代码的主要功能是从给定的源文件夹（s_path）中提取人脸图像，并将其保存到目标文件夹（face_dir）中。具体步骤如下：
# 导入所需的库，如os、numpy、cv2、YOLO和PIL。
# 定义一个名为create_path的函数，用于在给定路径上创建目录（如果不存在）。
# 定义一个名为process_image的函数，该函数接受一个YOLO对象、源文件路径（s_file）和目标文件路径（d_file）作为输入参数。
# 这个函数首先读取源文件中的图像，然后使用YOLO对象检测图像中的边界框（bbox）。如果检测到边界框，它将提取边界框内的图像区域，
# 将其大小调整为96x112像素，并将调整后的图像保存到目标文件中。
# 定义一个名为get_face_dataset的函数，该函数接受一个YOLO对象、源文件夹路径（s_path）
# 和目标文件夹路径（face_dir）作为输入参数。这个函数首先为每个子文件夹创建一个目录，然后遍历源文件夹中的所有文件。
# 对于每个文件，它调用process_image函数来处理图像并保存结果。
# 如果将此脚本作为主程序运行，它将创建一个名为face_dir的目录（如果尚不存在），
# 然后使用YOLO对象从名为'data/upload'的源文件夹中提取人脸图像，并将结果保存到face_dir目录中。
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_image(yolo_, s_file, d_file):
    image = cv2.imread(s_file)
    image = Image.fromarray(image)
    try:
        bbox = yolo_.get_bboxs(image)
        if bbox is not None:
            x1 = int(bbox["left"])
            y1 = int(bbox["top"])
            x2 = int(bbox["right"])
            y2 = int(bbox["bottom"])
            image = np.array(image, dtype='float32')
            image = image[y1:y2 + 1, x1:x2 + 1, :]
            image = cv2.resize(image, (96, 112))
            cv2.imwrite(d_file, image)
            # cv2.imencode('.jpg', image)[1].tofile(d_file)

    except Exception as ex:
        print('-------------------error-------------------------')
        print(ex)


def get_face_dataset(yolo_, s_path, face_dir):
    # 创建face_dir的目录
    for dir in os.listdir(s_path):
        dir_path = face_dir + '/' + dir
        create_path(dir_path)

    for root, dirs, names in os.walk(s_path):
        for filename in names:
            s_file = s_path + '/' + root.split('\\')[-1] + '/' + filename
            d_file = face_dir + '/' + root.split('\\')[-1] + '/' + filename
            process_image(yolo_, s_file, d_file)
            # print(d_file)

    print('done')


if __name__ == '__main__':
    # process_dataset(os.path.join('data', 'dataset'), os.path.join('data', 'result'))
    # 创建目录

    face_dir = 'data/face_dataset'
    create_path(face_dir)

    get_face_dataset(YOLO(), 'data/upload', face_dir)
