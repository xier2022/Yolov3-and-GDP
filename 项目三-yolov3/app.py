import colorsys
import os
import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.models import load_model
import settings
from yolo3.model_Mobilenet import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
# 人脸特征导入
from model_mobilenet.mobilenetv2_arcface import *
import cv2
import pickle
import uuid

from flask import Flask, render_template, request, jsonify,redirect, url_for
import base64
from io import BytesIO
from PIL import Image

from business import face_training,face_recognition
from flask_socketio import SocketIO
import base64

from flask_cors import CORS, cross_origin

# socketio = SocketIO(cors_allowed_origins="*")
socketio = SocketIO()
app = Flask(__name__)
CORS(app, resources=r'/*')
socketio.init_app(app, cors_allowed_origins="*")





@app.route('/train_model', methods=['GET'])
def trigger_train_model():
    try:
        face_training.train_model()
        return "Model trained successfully"  # 返回成功消息
    except Exception as e:
        return f"Error: {e}"  # 返回错误信息

@socketio.on('video_frame')
def process_frame(data):
    # 处理接收到的图像帧数据
    frame_data = base64.b64decode(data['frame'].split(',')[1])  # 解码接收到的图像帧数据
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 将接收到的图像帧解码为图像对象

    # 在图像帧上执行人脸检测或其他处理
    processed_frame = face_recognition.face_detect_demo(frame)

    # 将处理后的图像帧转成base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # 将处理后的图像帧发送回前端
    socketio.emit('processed_frame', {'processedFrame': 'data:image/jpeg;base64,' + processed_frame_base64})

# def invert_colors(frame):
#     # 图像处理逻辑示例：反转颜色
#     return frame

# 
# @app.route('/')
# def index():
#     return render_template('main.html')

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
        self.get_all_face_features()

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
        print(model_path)
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

    def get_all_face_features(self):
        # 加载人脸特征库文件
        with open("feature_database.pkl", "rb") as f:
            self.all_features = pickle.load(f)

    def get_single_face_feature(self, img):
        # 获取单个人脸图像的人脸特征
        img = self.__preprocess(img)
        embedding_img = self.face_feature_model.predict(img)
        embedding_img = embedding_img / np.expand_dims(np.sqrt(np.sum(np.power(embedding_img, 2), 1)), 1)
        return embedding_img

    def save_single_face_feature(self, name, saved_embedding):
        self.all_features[name] = saved_embedding
        with open("feature_database.pkl", "wb") as f:
            pickle.dump(self.all_features, f)

    def del_single_face_feature(self, name):
        del self.all_features[name]
        with open("feature_database.pkl", "wb") as f:
            pickle.dump(self.all_features, f)

    def get_person_name(self, query_embedding, threshold=0.5):
        # 根据人脸特征，查找最相似的人, 返回姓名
        max = 0
        person_name = ""
        for name, feature in self.all_features.items():
            score = np.sum(np.multiply(feature, query_embedding), 1)
            if score > max:
                max = score
                person_name = name
        if max < threshold:
            person_name = ""
        print("max sim: ", max)
        return person_name

    @tf.function
    def compute_output(self, image_data, image_shape):
        # Generate output tensor targets for filtered bounding boxes.
        # self.input_image_shape = K.placeholder(shape=(2,))
        self.input_image_shape = tf.constant(image_shape)
        boxes, scores, classes = yolo_eval(self.yolo_model(image_data), self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def get_bboxs(self, image):
        """
        获取检测的人脸框(若有多个人脸，只返回最高置信度的人脸框)
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
        if len(out_boxes) == 0:
            return {}
        for i, c in reversed(list(enumerate(out_scores))):
            box = out_boxes[i]
            score = out_scores[i]
            return {"box": box, "score": score}


    def draw_box_image(self, image, data, raw=None, username=None):
        # image: 原图像
        # data: 检测到的人脸框及姓名
        #return :   result: True or False 表明是否在人脸库中, face_feature 人脸特征
        size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
        font = ImageFont.truetype(
            "font/msyh.ttc", size= 2*size, encoding="utf-8")
        print("font size: ", np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # 在图像上绘制边界框，分类、置信度
        box = data["box"]
        score = data["score"]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        if raw is None:
            face_data = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            face_data = raw
        face_image = face_data[top:bottom + 1, left:right + 1, :]
        face_image = cv2.resize(face_image, (96, 112))
        face_feature = self.get_single_face_feature(face_image)

        # 改进
        try:
            name = self.get_person_name(face_feature)
        except:
            name = ''
            print('获取姓名失败')

        # if not username:
        #     name = self.get_person_name(face_feature)
        # else:
        #     name = username
        if not name:
            name = "未知"
            print("不在人脸库中，请录入")
            result = False
        else:
            result = True
        label = '{}'.format(name)
        print(label, (left, top), (right, bottom))
        # ------------------图像绘制------------------#
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle( [left + i, top + i, right - i, bottom - i], outline=self.colors[0])
        draw.rectangle( [tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[0])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        del draw
        return result, image, face_feature, name

yolo = YOLO()

def base64_to_cvimage(frame):
    head, context= frame.split(",")
    image = base64.b64decode(context)
    image = Image.open(BytesIO(image))
    return image

def cvimage_to_base64(frame):
    # frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    # frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    _, image = cv2.imencode(".jpg", frame)
    image = image.tobytes()
    image = base64.b64encode(image).decode("ascii")
    return image

app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True

def get_path(username):
    uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, username))
    uid = uid.replace("-", "_")
    return "static/images/" + uid + ".jpg"

@app.route("/login", methods=["GET", "POST"])
def detect_face():
    print("正在登录")
    data = request.values.to_dict()
    frame = base64_to_cvimage(data["image"])
    bbox_data = yolo.get_bboxs(frame)
    if not bbox_data:
        return {"code": 400, "data":"", "msg":"未检测到人脸，请对准摄像头。"}
    result, r_image, face_feature,username = yolo.draw_box_image(frame, bbox_data)
    if not result:
        print("尚未注册，请先注册。")
        return {"code": 401, "data": "", "msg": "尚未注册。请先注册。"}
    img_path = get_path(username)
    if os.path.exists(img_path):
        os.remove(img_path)
    r_image = np.array(r_image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, r_image)
    return {"code":200, "data":img_path.split("/")[-1], "msg":"登录成功!", "username":username}

# @cross_origin(origin='http://127.0.0.1:5000')
# def login():
#     # 处理登录逻辑
#     return jsonify({'message': 'Login successful'})


@app.route("/register", methods=["GET", "POST"])
def register():
    print("正在注册")
    data = request.values.to_dict()
    frame = base64_to_cvimage(data["image"])
    username = data['username']
    bbox_data = yolo.get_bboxs(frame)
    if not bbox_data:
        return {"code": 400, "data":"", "msg":"未检测到人脸，请对准摄像头重新拍照"}
    result, r_image, face_feature, _ = yolo.draw_box_image(frame, bbox_data, username=username)

    if result:
        msg = "您已经注册过，将为您直接登录"
    else:
        msg = "注册成功！正在为您跳转。"
        yolo.save_single_face_feature(username, face_feature) # 保存特征
    img_path = get_path(username)
    if os.path.exists(img_path):
        os.remove(img_path)

    r_image = np.array(r_image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, r_image)

    return {"code":200, "data": img_path.split("/")[-1], "msg": msg, "username":username}

@app.route("/home_page/<username>/<path>", methods=["GET", "POST"])
def home_page(username, path):
    print("username, path: ", username, path)
    response=requests.get("http://127.0.0.1:5000")
    return render_template("new.html", username=username, img_path=path)


@app.route("/")
def index():
    return render_template("index.html",faceNames=list(yolo.all_features.keys()))
    # return render_template("index.html")

# 删除数据路由
@app.route('/del_feature', methods=['GET','POST'])
def del_feature():
    data = request.values.to_dict("data")
    name = data["name"]
    if name in yolo.all_features.keys():
        yolo.del_single_face_feature(name)  # 删除特征数据
        img_path = get_path(name)
        if os.path.exists(img_path):  # 删除图片
            os.remove(img_path)
        return {"code":200,"data":img_path.split("/")[-1],"msg":"删除成功"}
    else:
        return {"code":201,"data":"name","msg":"删除失败，请选择用户！"}

@app.route("/test", methods=['GET'])
def test():
    return render_template("test.html", img_path ="images/wht.jpg")

@app.route('/manage',methods=['GET'])
def manage():
    names = list(yolo.all_features.keys())
    facejson = []
    for name in names:
        img_path = get_path(name)
        facejson.append({"name":name,"img_path":img_path})
    return render_template("manage.html",face_list=facejson)


if __name__== "__main__":
    app.run(debug=True,use_reloader=False,port=5002)
