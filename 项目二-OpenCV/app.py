import cv2
import numpy as np
from flask import Flask, render_template
from business import face_training,face_recognition
from flask_socketio import SocketIO
import base64
socketio = SocketIO()
app = Flask(__name__)
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


@app.route('/')
def index():
    return render_template('main.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)
