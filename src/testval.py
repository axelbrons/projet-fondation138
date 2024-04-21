import cv2
import numpy as np
import cv2.aruco as aruco
import sys


# Chemin vers votre image contenant le marqueur ArUco
chemin_image = "..\\img\\carte3.jpg"
chemin_image1 = "..\\img\\carte4.jpg"
chemin_image2 = "..\\img\\carte5.jpg"
chemin_image3 = "..\\img\\carte9.jpg"





##Detection des arucos
def trouver_id_aruco(image_path, dictionary=aruco.DICT_ARUCO_ORIGINAL):
  # Charger l'image
  image_aruco = cv2.imread(image_path)
  cap= image_aruco
  # Vérifier si l'image a été chargée avec succès
  if image_aruco is None:
      print(f"Erreur : Impossible de charger l'image à partir de {image_path}")
      sys.exit()










  # Convertir l'image en niveaux de gris
  image_grayscale = cv2.cvtColor(image_aruco, cv2.COLOR_BGR2GRAY)
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
      # Dessiner les contours et les ID des marqueurs détectésp
      image_with_markers = aruco.drawDetectedMarkers(image_aruco.copy(), corners, ids)
      # Afficher l'image avec les marqueurs détectés
      cv2.imshow('Marqueurs ArUco', image_with_markers)

      cv2.waitKey(0)
      cv2.destroyAllWindows()
      # Retourner les IDs des marqueurs détectés
      return ids.flatten(), coin_H_G, coin_H_D, coin_B_D, coin_B_G
  else:
      print("Aucun marqueur ArUco n'a été détecté dans l'image.")
      return None




# Appeler la fonction pour trouver l'ID du marqueur ArUco


ids_marqueurs, coin_H_G, coin_H_D, coin_B_D, coin_B_G = trouver_id_aruco(chemin_image)



image1 = cv2.imread(chemin_image)

pts1 = np.float32([coin_H_G, coin_H_D, coin_B_D, coin_B_G])
pts2 = np.float32([[0, 0], [1680, 0], [1680, 945], [0, 945]])


M = cv2.getPerspectiveTransform(pts1, pts2)

warped = cv2.warpPerspective(image1, M, (1680, 945))




# Afficher les IDs des marqueurs détectés

if ids_marqueurs is not None:
  print("IDs des marqueurs détectés :", ids_marqueurs)


def detect_cubes_in_image(image_path, lo_rouge, hi_rouge, lo_gris, hi_gris, lo_bleu, hi_bleu, lo_vert, hi_vert,
                          lo_jaune, hi_jaune, lo_blanc, hi_blanc):
    # Charger l'image
    image = cv2.imread(image_path)

    # Convertir l'image en HSV et en niveaux de gris
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Créer les masques pour différentes couleurs
    mask_rouge = cv2.inRange(image_hsv, lo_rouge, hi_rouge)
    mask_gris = cv2.inRange(image_hsv, lo_gris, hi_gris)
    mask_bleu = cv2.inRange(image_hsv, lo_bleu, hi_bleu)
    mask_vert = cv2.inRange(image_hsv, lo_vert, hi_vert)
    mask_jaune = cv2.inRange(image_hsv, lo_jaune, hi_jaune)

    # Seuillage pour détecter les objets blancs
    _, thresholded_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)

    # Erosion et dilatation pour améliorer les masques
    mask_rouge = cv2.erode(mask_rouge, None, iterations=1)
    mask_rouge = cv2.dilate(mask_rouge, None, iterations=1)
    mask_gris = cv2.erode(mask_gris, None, iterations=1)
    mask_gris = cv2.dilate(mask_gris, None, iterations=1)
    mask_bleu = cv2.erode(mask_bleu, None, iterations=1)
    mask_bleu = cv2.dilate(mask_bleu, None, iterations=1)
    #mask_vert = cv2.erode(mask_vert, None, iterations=1)
    #mask_vert = cv2.dilate(mask_vert, None, iterations=1)
    #mask_jaune = cv2.erode(mask_jaune, None, iterations=3)
    #mask_jaune = cv2.dilate(mask_jaune, None, iterations=3)
    #thresholded_image = cv2.erode(thresholded_image, None, iterations=3)
    #thresholded_image = cv2.dilate(thresholded_image, None, iterations=3)

    # Détecter les contours et dessiner les objets détectés
    result_image = image.copy()
    for i in range(30):
        detect_and_draw(result_image, mask_rouge, "Cube rouge")
        detect_and_draw(result_image, mask_gris, "Cube gris")
        detect_and_draw(result_image, thresholded_image, "Cube blanc")

    for i in range(10):
        detect_and_draw(result_image, mask_bleu, "Cube bleu")

    for i in range(3):

        detect_and_draw(result_image, mask_vert, "Cube vert")
        detect_and_draw(result_image, mask_jaune, "Cube jaune")


    # Afficher l'image avec les objets détectés
    cv2.imshow('Detected Cubes', result_image)
    #cv2.imshow('mask_rouge', mask_rouge)
    cv2.imshow('mask-gris', mask_gris)
    #cv2.imshow('mask_bleu', mask_bleu)
    #cv2.imshow('mask_blanc', thresholded_image)
    #cv2.imshow('mask_vert', mask_vert)
    #cv2.imshow('mask_jaune', mask_jaune)
    #cv2.imshow('mask_blanc', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_and_draw(image, mask, label):
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c = max(elements, key=cv2.contourArea)
        ((x, y), rayon) = cv2.minEnclosingCircle(c)
        if rayon > 10:
            cv2.circle(mask, (int(x), int(y)), int(rayon), (0, 255, 255), 2)
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 255), 10)
            cv2.line(image, (int(x), int(y)), (int(x) + 150, int(y)), (0, 255, 255), 2)
            cv2.putText(image, label, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)


# Paramètres HSV pour les différentes couleurs
lo_rouge = np.array([0,100,100])
hi_rouge = np.array([10,255,255])
lo_gris = np.array([50,50,50])
hi_gris = np.array([150,150,150])
lo_bleu = np.array([100,100,100])
hi_bleu = np.array([130,200,200])
lo_vert = np.array([40,40,40])
hi_vert = np.array([80,255,255])
lo_jaune = np.array([20,100,100])
hi_jaune = np.array([30,255,255])
lo_blanc = np.array([200,200,200])
hi_blanc = np.array([255,255,255])
color_infos=(0,255,255)


# Appeler la fonction pour la détection des cubes dans l'image
detect_cubes_in_image(chemin_image, lo_rouge, hi_rouge, lo_gris, hi_gris, lo_bleu, hi_bleu, lo_vert, hi_vert, lo_jaune,
                      hi_jaune, lo_blanc, hi_blanc)
