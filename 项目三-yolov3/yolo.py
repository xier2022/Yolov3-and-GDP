# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model

import settings
from yolo3.model_Mobilenet import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import pickle
import json
# 人脸特征导入
from model_mobilenet.mobilenetv2_arcface import *
import matplotlib.pyplot as plt
import cv2


face_features = {}
# 初始化参数：设置模型路径、锚点路径、类别路径等。
class YOLO(object):
    _defaults = {
        "model_path": settings.DEFAULT_MODEL_PATH,
        "anchors_path": settings.DEFAULT_ANCHORS_PATH,
        "classes_path": settings.DEFAULT_CLASSES_PATH,
        "score": settings.SCORE,
        "iou": settings.IOU,
        "model_image_size": settings.MODEL_IMAGE_SIZE,
        "gpu_num": settings.GPU_NUM,
        "feature_path": settings.FACE_FEATURE_MODEL,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.load_yolo_model()
        self.load_face_feature_model()
        self.get_all_face_feature()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

    # 添加的人脸特征代码
    def load_face_feature_model(self):
        # 加载人脸识别模型
        self.face_feature_model = load_model(self.feature_path)

    def __preprocess(self, img):
        # 人像图像预处理
        img = (img.astype('float32') - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)
        return img

    def get_all_face_feature(self):
        # 处理图像库中的人脸，获取人脸特征
        self.face_list = []
        temp = {}
        for root, dirs, names in os.walk('data/face_dataset'):
            if len(names) > 0:
                for name in names:
                    image_path = root + '/' + name
                    img = self.__preprocess(plt.imread(image_path))
                    embedding_img = self.face_feature_model.predict(img)
                    embedding_img = embedding_img / np.expand_dims(np.sqrt(np.sum(np.power(embedding_img, 2), 1)), 1)
                    person = image_path.split('\\')[-1].split('/')[0]
                    self.face_list.append({'name': person, 'embedding': embedding_img})
                    if person not in temp:
                        temp[person] = embedding_img
        with open("feature_database.pkl", "wb") as f:
            pickle.dump(temp, f)


    def get_detect_name(self, image_path):
        # 将人脸与图像库中的人脸一一比对，获取相似度最高的人脸及名称
        print(image_path)
        img = self.__preprocess(plt.imread(image_path))
        embedding_test = self.face_feature_model.predict(img)
        embedding_test = embedding_test / np.expand_dims(np.sqrt(np.sum(np.power(embedding_test, 2), 1)), 1)
        max = 0
        name = ''
        for face in self.face_list:
            score = np.sum(np.multiply(face['embedding'], embedding_test), 1)
            if score > max:
                max = score
                name = face['name']

        if max < 0.5:
            name = '未知'  # '数据库中无此人'
        return name

    @tf.function
    def compute_output(self, image_data, image_shape):
        # Generate output tensor targets for filtered bounding boxes.
        # self.input_image_shape = K.placeholder(shape=(2,))
        self.input_image_shape = tf.constant(image_shape)
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model(image_data), self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def get_bboxs(self, image):
        """
        获取检测的人脸框
        :param image:
        :return:
        """
        # 图像预处理
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 调用compute_output获取边界框、置信度、分类
        out_boxes, out_scores, out_classes = self.compute_output(image_data, [image.size[1], image.size[0]])
        bboxs = []
        maxarea = 0
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            bbox = dict([("score", str(score)), ("left", str(left)),
                         ("top", str(top)), ("right", str(right)), ("bottom", str(bottom))])
            # area = (x2 - x1) * (y2 - y1)
            # if area >= maxarea:
            #     maxarea = area
            bboxs.append(bbox)
            return bbox

    def detect_image(self, image, raw=None):
        """
        检测图片
        :param image: 图片
        :return:
        """
        start = timer()  # 开始时间
        # 图像预处理
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 调用compute_output获取边界框、置信度、分类
        out_boxes, out_scores, out_classes = self.compute_output(image_data, [image.size[1], image.size[0]])

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #                           size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 设置显示的字体，可改成中文
        font = ImageFont.truetype(
            "font/msyh.ttc", size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'), encoding="utf-8")
        thickness = (image.size[0] + image.size[1]) // 300
        # 在图像上绘制边界框，分类、置信度
        print("****************",out_scores.shape, out_scores.numpy())
        print('****************', reversed(list(enumerate(out_scores))))
        print('****************', reversed(list(enumerate(out_scores.numpy()))))

        index = 0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            print("score***************: ",score)
            # label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # 裁剪人像，获取name
            if raw is None:
                face_data = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                face_data = raw
            face_data = face_data[top:bottom + 1, left:right + 1, :]
            face_data = cv2.resize(face_data, (96, 112))
            face_path = './data/test1/test' + str(index) + '.jpg'
            cv2.imwrite(face_path, face_data)
            name = self.get_detect_name(face_path)
            # label = '{} {:.2f}'.format(name, score)
            label = '{}'.format(name)
            print(label, (left, top), (right, bottom))
            # ------------------图像绘制------------------#
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image


def detect_video(yolo, video_path="", output_path=""):
    import cv2
    if video_path == "":
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image, frame)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # image
    # test_path = './data/test/zt01.jpg'
    # image = Image.open(test_path)
    # image=YOLO().detect_image(image)
    # image.show()
    # video
    # detect_video(YOLO(), './data/test/double.mp4')
    detect_video(YOLO())
