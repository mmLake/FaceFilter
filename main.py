import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

HAARCASCADES_PATH = "/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
DEFAULT_CAMERA_IDX = 0

#opens default camera
camera = cv.VideoCapture(DEFAULT_CAMERA_IDX)

while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame',gray)

    # load cascade classifier training file for haarcascade
    haar_face_cascade = cv.CascadeClassifier(HAARCASCADES_PATH)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = haar_face_cascade.detectMultiScale(gray, 1.1, 5);

    # print the number of faces found
    print('Faces found: ', len(faces))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv.destroyAllWindows()