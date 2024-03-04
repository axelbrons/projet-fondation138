import cv2
import numpy as np
import imutils


def ouvrir_image(image):
    cv2.namedWindow('image de base', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0