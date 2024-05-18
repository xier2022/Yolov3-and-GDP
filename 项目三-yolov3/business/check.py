import cv2
import os



recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'../trained_model.yml')

names=[]

def name():
    #global names
    path = 'C:/Users/西耳/Downloads/'
    # names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)
    return names

def face_detect_demo(img):
    #img=cv2.imread('save/2.face.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector=cv2.CascadeClassifier('../haarcascade_frontalface_alt2.xml')
    face=face_detector.detectMultiScale(gray,1.1,5)

    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        #cv2.circle(gray,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)

        ids,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # print('id:',ids)
        # print('id：',ids,'置信评分：',confidence)
        if confidence<80:
            cv2.putText(img, str(names[ids]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)

        else:
            cv2.putText(img, 'unkonwn', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result',img)


names = name()

cap=cv2.VideoCapture(0)
while True:
    flag,frame=cap.read()
    # if not flag:
    #     break
    face_detect_demo(frame)
    if ord(' ')==cv2.waitKey(1):
        break
cv2.destroyAllWindows()
cap.release()