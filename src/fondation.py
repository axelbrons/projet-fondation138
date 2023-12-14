import cv2
import numpy as np
import imutils
# colors
from webcolors import rgb_to_name, CSS3_HEX_TO_NAMES, hex_to_rgb  # python3 -m pip install webcolors
from scipy.spatial import KDTree


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


def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES  # css3_hex_to_names
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        elif len(approx) == 10 or len(approx) == 12:
            shape = "star"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


if __name__ == '__main__':
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while (True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # load the image and resize it to a smaller factor so that
        # the shapes can be approximated better
        resized = imutils.resize(frame, width=300)
        ratio = frame.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()
        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)

            # Vérifier si M["m00"] n'est pas égal à zéro avant la division
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)

            # detect shape from contour
            shape = sd.detect(c)

            # resize the contour
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            # draw contour with mask
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Convert to RGB and get color name
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean = cv2.mean(imgRGB, mask=mask)[:3]
            named_color = convert_rgb_to_names(mean)

            # get complementary color for text
            mean2 = (255 - mean[0], 255 - mean[1], 255 - mean[2])

            # display shape name and color
            objLbl = shape + " {}".format(named_color)
            textSize = cv2.getTextSize(objLbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(frame, objLbl, (int(cX - textSize[0] / 2), int(cY + textSize[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, mean2, 2)

            # show image
            cv2.imshow("Image", frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
