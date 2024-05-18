import cv2  # 导入OpenCV库
import numpy as np  # 导入numpy库，用于数组操作
import os  # 导入os库，用于文件路径操作
from PIL import Image  # 导入PIL库，用于图像处理



def getImageAndLabels(path):
    # 储存人脸数据
    facesSamples = []
    # 存储姓名数据
    ids = []
    # 储存图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if
                  f.lower() != "desktop.ini"]  # 获取指定路径下除了desktop.ini之外的所有文件路径
    # 加载人脸检测器分类器
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # 遍历列表中的图片
    for imagePath in imagePaths:

        # 打开图片并转换为灰度图
        pil_img = Image.open(imagePath).convert('L')
        # 使用PIL库中的Image.open()
        # 函数打开图像文件，然后通过.convert('L')
        # 将图像转换为灰度图像，这是因为人脸检测通常在灰度图像上进行。

        # 将图片转换为numpy数组
        img_numpy = np.array(pil_img, 'uint8')
        # PIL_img：是一个PIL图像对象，通过Image.open()
        # 函数打开并转换为灰度图像。
        # np.array()：是NumPy库中的函数，用于将输入转换为NumPy数组。
        # 'uint8'：是指定转换后的数组的数据类型。在这里，'uint8'
        # 表示无符号8位整数，范围从0到255。这是因为灰度图像的像素值通常在0到255之间，用8位表示。

        # 获取图片中的人脸特征
        faces = face_detector.detectMultiScale(img_numpy, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
        # detectMultiScale()函数返回一个包含人脸位置和尺寸的矩形列表。
        # img_numpy：是输入的图像，这里是一个NumPy数组，通常是灰度图像。这个函数会在这个图像中进行人脸检测。
        # scaleFactor：用于指定在图像金字塔中每个图像缩放比例的参数。它影响了算法在不同尺度上检测物体的能力。
        # 如果设置为1.2，表示每次缩小图像的比例为1.2倍。
        # minNeighbors：指定每个候选矩形应该保留多少个相邻矩形。这个参数可以用来调节检测结果的准确性。较高的值会导致更少的检测结果，但可能会提高准确性。
        # minSize：指定检测到的对象的最小尺寸。它是一个元组，包含了宽度和高度的最小值。在这里，设置为(30, 30)
        # 表示检测到的人脸矩形的宽度和高度至少为30个像素。

        # 获取图片的id和姓名
        id = os.path.split(imagePath)[1].split('.')[0]
        # os.path.split(imagePath)：这个函数将给定的路径分割成目录路径和文件名两部分，并返回一个元组。
        # 在这里，imagePath是图像文件的路径，所以os.path.split(imagePath)返回一个包含目录路径和文件名的元组。
        # [1]：这个索引表示我们只关心元组中的第二个元素，也就是文件名部分。
        # split('.')：这个方法根据指定的分隔符（这里是.）将字符串分割成多个部分，并返回一个包含分割后的字符串的列表。
        # [0]：这个索引表示我们只关心分割后的列表中的第一个元素，也就是文件名中的ID部分。

        for x, y, w, h in faces:
            ids.append(id)  # 将id添加到列表中
            facesSamples.append(img_numpy[y:y + h, x:x + w])  # 将人脸样本添加到列表中
            # 在这段代码中，faces是一个存储检测到的人脸位置信息的列表。
            # 每个元素都是一个包含四个值的元组，分别表示检测到的人脸的位置和尺寸：x坐标、y坐标、宽度（width）、高度（height）。
            # 在for x, y, w, h in faces: 循环中，对于每个检测到的人脸，代码会执行以下操作：
            # x, y, w, h：这些变量分别接收了当前循环中人脸位置信息元组中的四个值，即人脸矩形的左上角顶点的 x 和 y 坐标，以及矩形的宽度和高度。
            # ids.append(id)：将之前从图像文件路径中提取出的ID添加到名为 ids 的列表中。这个列表用于存储每个人脸对应的ID。
            # facesSamples.append(img_numpy[y:y + h, x:x + w])：从原始图像数组img_numpy中裁剪出当前检测到的人脸区域，
            # 并将裁剪出的人脸样本添加到名为facesSamples的列表中。这样，facesSamples列表就存储了所有检测到的人脸样本。
            #
            # 所以，faces是一个包含了检测到的人脸位置信息的列表，而forx, y, w, h in faces: 循环则用于遍历每个检测到的人脸，并对其进行后续处理。

    print('id:', ids)  # 打印人脸id信息
    print('fs:', facesSamples)  # 打印人脸样本信息
    return facesSamples, ids  # 返回人脸样本和id列表


def train_model():
    path = 'D/paper/xfei/face_id/test/face/'  # 图片文件路径
    if not os.path.exists(path):  # 判断路径是否存在
        return "Error: The specified path does not exist."

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPHFaceRecognizer识别器
        faces, ids = getImageAndLabels(path)  # 获取人脸样本和id列表
        # umat_ids = convert_to_umat(ids)
        recognizer.train(faces, np.array(ids, dtype=np.int32))  # 训练模型

        recognizer.write('D:/paper/xfei/face_opencv/face_opencv/flaskProject/trained_model.yml')  # 保存训练好的模型
        return "Model trained successfully"
    except Exception as e:
        return f"Error: {e}"  # 返回错误信息


# # 主函数
# if __name__ == '__main__':
#     path = 'C:/Users/西耳/Downloads/'  # 图片文件路径
#     if not os.path.exists(path):  # 判断路径是否存在
#         print("Error: The specified path does not exist.")
#     else:
#         try:
#             recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPHFaceRecognizer识别器
#             faces, ids = getImageAndLabels(path)  # 获取人脸样本和id列表
#             # umat_ids = convert_to_umat(ids)
#             recognizer.train(faces, np.array(ids, dtype=np.int32))  # 训练模型
#
#             recognizer.write('D:/face_opencv/flaskProject/trained_model.yml')  # 保存训练好的模型
#         except Exception as e:
#             print("Error:", e)  # 打印错误信息
