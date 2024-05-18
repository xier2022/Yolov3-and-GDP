import cv2
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64

def capture_images(save_path="D:/face_opencv/flaskProject/images/", name="yyh"):
    capture = cv2.VideoCapture(0)
    num = 0



    while capture.isOpened():
        flag, show = capture.read()
        cv2.imshow("Capture_Test", show)
        k = cv2.waitKey(1) & 0xff

        if k == ord('s'):
            cv2.imwrite(os.path.join(save_path, str(num) + "." + name + ".jpg"), show)
            print("success to save" + str(num) + ".jpg")
            print("----------------------")
            num += 1
        elif k == ord(' '):
            break

    capture.release()
    cv2.destroyAllWindows()

# 调用示例
# capture_images()  # 默认保存路径和名称为"yyh"
# capture_images(save_path="your_custom_path", name="your_custom_name")
