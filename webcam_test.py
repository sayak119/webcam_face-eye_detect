import cv2
import sys
import datetime as dt
import logging as log
from time import sleep

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
log.basicConfig(filename='webcam.log',level=log.INFO)
video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Webcam loading failed.... Try Again Later !!!')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]
    	eyes = eyeCascade.detectMultiScale(roi_gray)
    	for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    
    # Logging into webcam.log (stackoverflow) error checking method
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" and eyes: "+str(len(eyes))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finally, release the capture
video_capture.release()
cv2.destroyAllWindows()
