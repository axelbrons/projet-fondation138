# import cv2
# import cv2.aruco as aruco
# import sys
#
# ##Detection des arucos :
#
# def trouver_id_aruco(image_path, dictionary=aruco.DICT_ARUCO_ORIGINAL):
#     # Charger l'image
#     image = cv2.imread(image_path)
#
#     # Vérifier si l'image a été chargée avec succès
#     if image is None:
#         print(f"Erreur : Impossible de charger l'image à partir de {image_path}")
#         sys.exit()
#
#     # Convertir l'image en niveaux de gris
#     image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Obtenir le dictionnaire ArUco prédéfini
#     aruco_dict = aruco.getPredefinedDictionary(dictionary)
#
#     # Initialiser les paramètres du détecteur ArUco
#     parameters = aruco.DetectorParameters()
#
#     # Détecter les coins des marqueurs
#     corners, ids, rejected_img_points = aruco.detectMarkers(image_grayscale, aruco_dict, parameters=parameters)
#
#     if ids is not None:
#         # Dessiner les contours et les ID des marqueurs détectés
#         image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
#
#         # Afficher l'image avec les marqueurs détectés
#         cv2.imshow('Marqueurs ArUco', image_with_markers)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         # Retourner les IDs des marqueurs détectés
#         return ids.flatten()
#     else:
#         print("Aucun marqueur ArUco n'a été détecté dans l'image.")
#         return None
#
#
# # Chemin vers votre image contenant le marqueur ArUco
# chemin_image = '..\\img\\carte3.jpg'
#
#
# # Appeler la fonction pour trouver l'ID du marqueur ArUco
# ids_marqueurs = trouver_id_aruco(chemin_image)
#
# # Afficher les IDs des marqueurs détectés
# if ids_marqueurs is not None:
#     print("IDs des marqueurs détectés :", ids_marqueurs)
#
#
 # import cv2
 # from cv2 import aruco
 # import sys
 #
 # def detecter_arucos_en_direct(dictionary=aruco.DICT_ARUCO_ORIGINAL):
 #     # Ouvrir la capture vidéo depuis la webcam (id=0 pour la webcam par défaut)
 #     cap = cv2.VideoCapture(0)
 #
 #     while True:
 #         # Lire un frame depuis la webcam
 #         ret, frame = cap.read()
 #
 #         # Convertir le frame en niveaux de gris
 #         frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #
 #         # Obtenir le dictionnaire ArUco prédéfini
 #         aruco_dict = aruco.getPredefinedDictionary(dictionary)
 #
 #         # Initialiser les paramètres du détecteur ArUco
 #         parameters = aruco.DetectorParameters()
 #
 #         # Détecter les coins des marqueurs
 #         corners, ids, rejected_img_points = aruco.detectMarkers(frame_grayscale, aruco_dict, parameters=parameters)
 #
 #         if ids is not None:
 #             # Dessiner les contours et les ID des marqueurs détectés
 #             frame_with_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
 #
 #             # Afficher l'image avec les marqueurs détectés
 #             cv2.imshow('Marqueurs ArUco', frame_with_markers)
 #         else:
 #             # Afficher le frame original s'il n'y a pas de marqueur détecté
 #             cv2.imshow('Marqueurs ArUco', frame)
 #
 #         # Sortir de la boucle si la touche 'q' est pressée
 #         if cv2.waitKey(1) & 0xFF == ord('q'):
 #             break
 #
 #     # Libérer la capture vidéo et fermer les fenêtres
 #     cap.release()
 #     cv2.destroyAllWindows()
 #
 # if __name__ == "__main__":
 #     # Appeler la fonction pour détecter les ArUcos depuis la webcam
 #     detecter_arucos_en_direct()

# """
# import cv2
# import numpy as np
# import imutils
#
# class ShapeDetector:
#     def detect(self, c):
#         # Initialise le nom de la forme et l'approximation du contour
#         shape = "Unknown"
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.09 * peri, True)
#
#         # Si le contour a 3 côtés, c'est un triangle
#         if len(approx) == 3:
#             shape = "Triangle"
#         # Si le contour a 4 côtés, c'est un carré ou un rectangle
#         elif len(approx) == 4:
#             (x, y, w, h) = cv2.boundingRect(approx)
#             aspect_ratio = w / float(h)
#
#             # On considère que c'est un carré si l'aspect ratio est proche de 1
#             shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
#         # Si le contour a 5 côtés, c'est un pentagone
#         elif len(approx) == 5:
#             shape = "Pentagon"
#         # Sinon, c'est probablement un cercle
#         else:
#             shape = "Circle"
#
#         return shape
#
# # Charger l'image
# image = cv2.imread("..\\img\\carte3.jpg") #nom de l'image (n'importe laquelle avec des formes
# resized = imutils.resize(image, width=300)
# ratio = image.shape[0] / float(resized.shape[0])
#
# # Convertir l'image redimensionnée en niveaux de gris, la flouter
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
# # Appliquer le seuillage adaptatif
# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
#
# # Filtrer les contours par aire
# seuil_min = 100
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = [c for c in cnts if cv2.contourArea(c) > seuil_min]
#
# # Appliquer un filtre morphologique (fermeture)
# kernel = np.ones((5, 5), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
# # Trouver les contours dans l'image seuillée
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# sd = ShapeDetector()
#
# # Boucle sur les contours détectés
# for c in cnts:
#     # Calculer le centre du contour
#     M = cv2.moments(c)
#     if (M["m00"]!=0) :
#         cX = int((M["m10"] / M["m00"]) * ratio)
#         cY = int((M["m01"] / M["m00"]) * ratio)
#     else :
#         cX = 0
#         cY =0
#     # Détecter la forme en utilisant le contour
#     shape = sd.detect(c)
#
#     # Multiplier les coordonnées (x, y) du contour par le ratio de redimensionnement
#     c = c.astype("float")
#     c *= ratio
#     c = c.astype("int")
#
#     # Dessiner les contours et le nom de la forme sur l'image
#     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#     cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
# # Afficher l'image résultante
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# """
#
#
# import cv2
#
# def calibrer_blancs(image_path):
#     # Charger l'image
#     image = cv2.imread(image_path)
#
#     # Convertir l'image de BGR à RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Appliquer l'équilibre automatique des blancs
#     wb = cv2.xphoto.createSimpleWB()
#     image_calib = wb.balanceWhite(image_rgb)
#
#     # Afficher l'image originale et l'image calibrée
#     # cv2.imshow('Image originale', image_rgb)
#     cv2.imshow('Image calibrée', cv2.cvtColor(image_calib, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # Spécifier le chemin de l'image
# chemin_image = "..\\img\\carte3.jpg"
#
# # Appeler la fonction de calibrage des blancs avec le chemin spécifié
# calibrer_blancs(chemin_image)


# import numpy as np
# import cv2
#
# img = cv2.imread('..\\img\\carte1.jpg')
# imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
# contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
#     cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1] - 5
#     if len(approx) == 3:
#         cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
#     elif len(approx) == 4 :
#         x, y , w, h = cv2.boundingRect(approx)
#         aspectRatio = float(w)/h
#         print(aspectRatio)
#         if aspectRatio >= 0.95 and aspectRatio < 1.05:
#             cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#         else:
#             cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#     elif len(approx) == 5 :
#         cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 10 :
#         cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     else:
#         cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
# cv2.imshow('shapes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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

import cv2
import cv2.aruco as aruco
import sys


from email.mime import image

import cv2
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

import cv2
import cv2.aruco as aruco
import sys

##Detection des arucos
def trouver_id_aruco(image_path, dictionary=aruco.DICT_ARUCO_ORIGINAL):
   # Charger l'image
   image = cv2.imread(image_path)
   # Vérifier si l'image a été chargée avec succès
   if image is None:
       print(f"Erreur : Impossible de charger l'image à partir de {image_path}")
       sys.exit()
   # Convertir l'image en niveaux de gris
   image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   # Obtenir le dictionnaire ArUco prédéfini
   aruco_dict = aruco.getPredefinedDictionary(dictionary)
   # Initialiser les paramètres du détecteur ArUco
   parameters = aruco.DetectorParameters()
   # Détecter les coins des marqueurs
   corners, ids, _ = aruco.detectMarkers(image_grayscale, aruco_dict, parameters=parameters)


   # Identifiez les marqueurs correspondant aux coins de la carte (par exemple, avec les ID 0, 1, 2 et 3)
   corner_ids = [0, 1, 2, 3]
   corner_markers = [marker for marker, marker_id in zip(corners, ids) if marker_id in corner_ids]

   # Extraire les coordonnées des coins des marqueurs identifiés
   corner_coordinates = []

   for marker, marker_id in zip(corners, ids):

       # Si l'ID du marqueur est 0, stocker les coordonnées du coin en haut à gauche
       if marker_id == 0:
           coin_H_G = marker[0][0]  # Coin en haut à gauche

       # Si l'ID du marqueur est 1, stocker les coordonnées du coin en haut à droite
       elif marker_id == 1:
           coin_H_D = marker[0][1]  # Coin en haut à droite

       # Si l'ID du marqueur est 2, stocker les coordonnées du coin en bas à droite
       elif marker_id == 2:
           coin_B_D = marker[0][2]  # Coin en bas à droite

       # Si l'ID du marqueur est 3, stocker les coordonnées du coin en bas à gauche
       elif marker_id == 3:
           coin_B_G = marker[0][3]  # Coin en bas à gauche

       for corner in marker[0]:
           corner_coordinates.append(corner)



   ## Afficher les coordonnées des coins
   #print("Coordonnées des coins des marqueurs ArUco identifiés comme des coins de la carte :")
   #for corner in corner_coordinates:
   #    print(corner)




   # Afficher les coordonnées du coin en haut à gauche si le marqueur avec ID 0 a été détecté
   if coin_H_G is not None:
       print("Coordonnées du coin en haut à gauche du marqueur ArUco avec ID 0 :", coin_H_G)
       coin_H_G -= 15
       print("Coordonnées du coin en haut à gauche du marqueur ArUco avec ID 0 :", coin_H_G)


   print("")

   # Afficher les coordonnées du coin en haut à droite si le marqueur avec ID 1 a été détecté
   if coin_H_D is not None:
       print("Coordonnées du coin en haut à droite du marqueur ArUco avec ID 1 :", coin_H_D)
       coin_H_D[0] += 15
       coin_H_D[1] -= 15
       print("Coordonnées du coin en haut à droite du marqueur ArUco avec ID 1 :", coin_H_D)

   print("")

   # Afficher les coordonnées du coin en bas à droite si le marqueur avec ID 2 a été détecté
   if coin_B_D is not None:
       print("Coordonnées du coin en bas à droite du marqueur ArUco avec ID 2 :", coin_B_D)
       coin_B_D += 25
       print("Coordonnées du coin en bas à droite du marqueur ArUco avec ID 2 :", coin_B_D)

   print("")

   # Afficher les coordonnées du coin en bas à gauche si le marqueur avec ID 3 a été détecté
   if coin_B_G is not None:
       print("Coordonnées du coin en bas à gauche du marqueur ArUco avec ID 3 :", coin_B_G)
       coin_B_G[0] -= 25
       coin_B_G[1] += 25
       print("Coordonnées du coin en bas à gauche du marqueur ArUco avec ID 3 :", coin_B_G)


   print("")


   if ids is not None:
       # Dessiner les contours et les ID des marqueurs détectés
       image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
       # Afficher l'image avec les marqueurs détectés
       cv2.imshow('Marqueurs ArUco', image_with_markers)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
       # Retourner les IDs des marqueurs détectés
       return ids.flatten(), coin_H_G, coin_H_D, coin_B_D, coin_B_G
   else:
       print("Aucun marqueur ArUco n'a été détecté dans l'image.")
       return None



def calibrer_blancs(image):
   # Convertir l'image de BGR à RGB
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # Appliquer l'équilibre automatique des blancs
   wb = cv2.xphoto.createSimpleWB()
   image_calib = wb.balanceWhite(image_rgb)
   # Afficher l'image originale et l'image calibrée
   # cv2.imshow('Image originale', image_rgb)
   cv2.imshow('Image calibrée', cv2.cvtColor(image_calib, cv2.COLOR_RGB2BGR))
   cv2.waitKey(0)
   cv2.destroyAllWindows()




# Chemin vers votre image contenant le marqueur ArUco

chemin_image = "..\\img\\carte3.jpg"


# Appeler la fonction pour trouver l'ID du marqueur ArUco

ids_marqueurs, coin_H_G, coin_H_D, coin_B_D, coin_B_G = trouver_id_aruco(chemin_image)



image1 = cv2.imread(chemin_image)

pts1 = np.float32([coin_H_G, coin_H_D, coin_B_D, coin_B_G])
pts2 = np.float32([[0, 0], [1680, 0], [1680, 945], [0, 945]])


M = cv2.getPerspectiveTransform(pts1, pts2)


warped = cv2.warpPerspective(image1, M, (1680, 945))


# Appeler la fonction de calibrage des blancs avec le chemin spécifié
calibrer_blancs(warped)


# Afficher les IDs des marqueurs détectés

if ids_marqueurs is not None:
   print("IDs des marqueurs détectés :", ids_marqueurs)

