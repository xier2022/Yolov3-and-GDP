from yolo import YOLO

import numpy as np
from PIL import Image


# yolo = YOLO()

def detect_video(yolo: YOLO):
    import cv2
    vid = cv2.VideoCapture(0)
    
    while True:
        ret, frame = vid.read()
        
        image = Image.fromarray(frame)
        image = yolo.detect_image(image, frame)
        result = np.asarray(image)
        
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_frame():
    ...
    
if __name__ == '__main__':
    # detect_video(yolo)
    ...
    