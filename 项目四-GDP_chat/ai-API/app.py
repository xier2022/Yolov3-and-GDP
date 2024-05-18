from flask import Flask, render_template
from flask_socketio import SocketIO
from business import api_call

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('main.html')


@socketio.on('message_from_frontend')
def handle_message(message):
    print('Received message from frontend:', message)

    response=api_call.spark_api(message)
    print('Response from api_call:', response)

    # 这里可以对接收到的消息进行处理，然后发送响应给前端
    # response = "Received: " + message
    socketio.emit('message_from_backend', response)  # 发送响应给前端


# @socketio.on('my_event')
# def handle_my_custom_event(json):
#     print('收到消息: ' + str(json))
#     socketio.emit('my_response', {'data': '这是服务器的响应!'})

if __name__ == '__main__':
    socketio.run(app,port=7000)
