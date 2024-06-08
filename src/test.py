
import cv2
import numpy as np
import cv2.aruco as aruco
import sys

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

while True:
    ret, frame=cap.read()
    chemin_image1= frame
    cv2.imshow('Image calibrée', frame)
    if cv2.waitKey(1)== ord('q'):
        break

cv2.destroyAllWindows()