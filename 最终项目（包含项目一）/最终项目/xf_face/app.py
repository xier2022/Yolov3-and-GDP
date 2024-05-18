import os
import json

import requests
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

from business.p1_face.face_compare import FaceCompare
from business.p1_face.face_feature_analysis import FaceFeature
from business.p1_face import face3, sex, score, expression
from business.p1_face.age import Age

UPLOAD_FOLDER = r"./p1_picture/"

# UPLOAD_FEATURE_IMAGE = "1.png"
# UPLOAD_COMPARE1_IMAGE = "1.png"
# UPLOAD_COMPARE2_IMAGE = "2.png"
UPLOAD_FEATURE_IMAGE = UPLOAD_FOLDER + "feature.png"
UPLOAD_COMPARE1_IMAGE = UPLOAD_FOLDER + "compare1.png"
UPLOAD_COMPARE2_IMAGE = UPLOAD_FOLDER + "compare2.png"



app = Flask(__name__)  # 创建程序实例

CORS(app)


# app.config['UPLOAD_FEATURE_IMAGE'] = UPLOAD_FOLDER + UPLOAD_FEATURE_IMAGE
# app.config['UPLOAD_COMPARE1_IMAGE'] = UPLOAD_FOLDER + UPLOAD_COMPARE1_IMAGE
# app.config['UPLOAD_COMPARE2_IMAGE'] = UPLOAD_FOLDER + UPLOAD_COMPARE2_IMAGE

# uploaded_image_data = None  # 全局变量用于保存图片地址


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/call', methods=['POST'])
def upload_file():
    res = requests.post('http://127.0.0.1:5002/login',data=request.files)
    return jsonify({"status": res.status_code, "data": res.text})

@app.route('/callzhuce', methods=['POST'])
def upload_file2():
    res = requests.post('http://127.0.0.1:5002/register')
    return jsonify({"status": res.status_code, "data": res.text})


@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_image_path
    global uploaded_image_name

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 保存图片路径和名称到全局变量
    uploaded_image_path = os.path.join(app.config['UPLOAD_FEATURE_IMAGE'], file.filename)
    uploaded_image_name = file.filename

    # 保存图片到指定路径
    file.save(uploaded_image_path)

    # 输出文件路径到控制台
    print("Uploaded file path:", uploaded_image_path)

    return jsonify({'message': 'File uploaded successfully'})


# 配置路由，用于显示第一个HTML页面
@app.route('/one')
def show_one():
    return render_template('p1_face_compare.html')

@app.route('/new/')
def show_new():
    return render_template('new.html')



@app.route('/four')
def show_four():
    return render_template('chat.html')


@app.route('/feature_compare', methods=['POST'])
def feature_compare():
    appid = 'd69bb9eb'
    api_secret = 'OTBiYjlhOGNmZjNhMmY5NTA2YWMyMGQ0'
    api_key = '34015cc2499b6b4d6dd493405d373e84'

    # 检查请求中是否包含两个文件
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Please provide two files'})

    file1 = request.files['file1']
    file2 = request.files['file2']

    # 保存文件到服务器
    file1_path = UPLOAD_COMPARE1_IMAGE
    file2_path = UPLOAD_COMPARE2_IMAGE
    file1.save(file1_path)
    file2.save(file2_path)

    # 调用处理图片的函数，这里使用您提到的 face3.run()，注意修改为您实际的处理函数
    ret = face3.run(appid, api_key, api_secret, file1_path, file2_path)

    # 返回处理结果
    return jsonify(json.loads(ret))


# 配置路由，用于显示第二个HTML页面
@app.route('/two')
def show_two():
    return render_template('p1_face_feature.html')


@app.route('/three')
def show_three():
    return render_template('opencv.html')


@app.route('/denglu')
def show_denglu():
    return render_template('denglu.html')


@app.route('/age', methods=['POST'])
def age():
    """
    人脸年龄识别
    :return:
    """
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 接受前端formdata传递过来的图像
    image_data = request.files['file'].save(UPLOAD_FEATURE_IMAGE)

    # 调用讯飞api
    age = Age(UPLOAD_FEATURE_IMAGE, 'feature.png')
    ret = age.run()

    # 返回条用结果
    return jsonify({"ret": ret})


@app.route('/score', methods=['POST'])
def score():
    """
    人脸年龄识别
    :return:
    """
    ret = score.run()
    return jsonify(json.loads(ret))


@app.route('/sex', methods=['POST'])
def sex():
    """
    人脸年龄识别
    :return:
    """
    ret = sex.run()
    return jsonify(json.loads(ret))


@app.route('/expresssion', methods=['POST'])
def expression():
    """
    人脸年龄识别
    :return:
    """
    ret = expression.run()
    return jsonify(json.loads(ret))


if __name__ == '__main__':

    app.run()
