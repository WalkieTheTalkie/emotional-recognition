import cv2
from deepface import DeepFace
import numpy as np

face_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'

face = cv2.CascadeClassifier()
if not face.load(cv2.samples.findFile(face_name)):
    print("error xml file")

video = cv2.VideoCapture(1)
print("hello") #test if code is still being read
while video.isOpened():
    _,frame = video.read() #uses the video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #turns video gray

    faces = face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5) #detecs faces

    for x,y,w,h in faces:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1) #creates the tracking rectangle


        try:
            analyze = DeepFace.analyze(frame,actions=['emotion'])#analyzes emotion

            print(analyze['dominant_emotion'])

        except:
            print("no face")
    
        cv2.imshow('video', frame)#displays output to user
        key=cv2.waitKey(1)
        if key==ord('q'):
            
            break

video.release()