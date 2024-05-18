from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('demo.html')

@socketio.on('video_frame')
def process_frame(data):
    # 处理接收到的图像帧数据
    processed_frame = invert_colors(data['frame'])
    # 将处理后的图像帧发送回前端
    socketio.emit('processed_frame', {'processedFrame': processed_frame})

def invert_colors(frame):

    return frame

if __name__ == '__main__':
    socketio.run(app)
