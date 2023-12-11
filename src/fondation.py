import cv2 as cv
import numpy as np

# Choix au hasard (zone de départ au hasard au debut de la game
# Capable d'aller chercher des objets de la couleur correspondante dans la zone de depart
# Ne pas depasser de la map
# Reconnaitre les couleurs
# Reconnaissance des objets sur la map
# Arret du robot en cas de panne de communication
# Permet d'arreter à tout moment le robot
# Doit en aucun cas rentrer en collision avec un mur ou un objet
# conversion pixels en mm (trouver la distance de la map)
# Detecter les elements sur la map et les placer sur une map virtuelle

cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()