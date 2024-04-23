

import cv2
import numpy as np
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
      #coin_H_G -= 15
      print("Coordonnées du coin en haut à gauche du marqueur ArUco avec ID 0 :", coin_H_G)
  print("")
  # Afficher les coordonnées du coin en haut à droite si le marqueur avec ID 1 a été détecté
  if coin_H_D is not None:
      print("Coordonnées du coin en haut à droite du marqueur ArUco avec ID 1 :", coin_H_D)
      #coin_H_D[0] += 15
      #coin_H_D[1] -= 15
      print("Coordonnées du coin en haut à droite du marqueur ArUco avec ID 1 :", coin_H_D)
  print("")
  # Afficher les coordonnées du coin en bas à droite si le marqueur avec ID 2 a été détecté
  if coin_B_D is not None:
      print("Coordonnées du coin en bas à droite du marqueur ArUco avec ID 2 :", coin_B_D)
      coin_B_D -= 5
      print("Coordonnées du coin en bas à droite du marqueur ArUco avec ID 2 :", coin_B_D)
  print("")
  # Afficher les coordonnées du coin en bas à gauche si le marqueur avec ID 3 a été détecté
  if coin_B_G is not None:
      print("Coordonnées du coin en bas à gauche du marqueur ArUco avec ID 3 :", coin_B_G)
      coin_B_G[0] += 5
      coin_B_G[1] -= 5
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


##Detection des arucos
def trouver_id_aruco2(image, dictionary=aruco.DICT_ARUCO_ORIGINAL):
  # Charger l'image

  # Vérifier si l'image a été chargée avec succès
  coin_H_D = None
  coin_B_D = None
  coin_B_G = None
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
      #coin_H_G -= 15
      print("Coordonnées du coin en haut à gauche du marqueur ArUco avec ID 0 :", coin_H_G)
  print("")
  # Afficher les coordonnées du coin en haut à droite si le marqueur avec ID 1 a été détecté
  if coin_H_D is not None:
      print("Coordonnées du coin en haut à droite du marqueur ArUco avec ID 1 :", coin_H_D)
      #coin_H_D[0] += 15
      #coin_H_D[1] -= 15
      print("Coordonnées du coin en haut à droite du marqueur ArUco avec ID 1 :", coin_H_D)
  print("")
  # Afficher les coordonnées du coin en bas à droite si le marqueur avec ID 2 a été détecté
  if coin_B_D is not None:
      print("Coordonnées du coin en bas à droite du marqueur ArUco avec ID 2 :", coin_B_D)
      coin_B_D -= 5
      print("Coordonnées du coin en bas à droite du marqueur ArUco avec ID 2 :", coin_B_D)
  print("")
  # Afficher les coordonnées du coin en bas à gauche si le marqueur avec ID 3 a été détecté
  if coin_B_G is not None:
      print("Coordonnées du coin en bas à gauche du marqueur ArUco avec ID 3 :", coin_B_G)
      coin_B_G[0] += 5
      coin_B_G[1] -= 5
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




def detect_cubes_in_image(image_path, lo_rouge, hi_rouge, lo_gris, hi_gris, lo_bleu, hi_bleu, lo_vert, hi_vert,
                          lo_jaune, hi_jaune, lo_blanc, hi_blanc):
    # Charger l'image
    image = image_path

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
   # mask_gris = cv2.erode(mask_gris, None, iterations=1)
   # mask_gris = cv2.dilate(mask_gris, None, iterations=1)
    #mask_bleu = cv2.erode(mask_bleu, None, iterations=1)
    #mask_bleu = cv2.dilate(mask_bleu, None, iterations=1)
    #mask_vert = cv2.erode(mask_vert, None, iterations=1)
    #mask_vert = cv2.dilate(mask_vert, None, iterations=1)
    #mask_jaune = cv2.erode(mask_jaune, None, iterations=3)
    #mask_jaune = cv2.dilate(mask_jaune, None, iterations=3)
    #thresholded_image = cv2.erode(thresholded_image, None, iterations=3)
    #thresholded_image = cv2.dilate(thresholded_image, None, iterations=3)

    # Détecter les contours et dessiner les objets détectés
    result_image = image.copy()
    image_noire = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
    for i in range(30):
        image_noire=detect_and_draw(result_image, mask_rouge, "Cube rouge",image_noire,(0,0,255))
        image_noire=detect_and_draw(result_image, mask_gris, "Cube gris",image_noire,(150,150,150))
        image_noire=detect_and_draw(result_image, thresholded_image, "Cube blanc",image_noire,(255,255,255))

    for i in range(10):
        image_noire=detect_and_draw(result_image, mask_bleu, "Cube bleu",image_noire,(100,0,0))

    for i in range(3):

        image_noire=detect_and_draw(result_image, mask_vert, "Cube vert",image_noire,(0,200,0))
        #image_noire=detect_and_draw(result_image, mask_jaune, "Cube jaune",image_noire,(30,255,255))


    # Afficher l'image avec les objets détectés
    cv2.imshow('Detected Cubes', result_image)
    #cv2.imshow('mask_rouge', mask_rouge)
    #cv2.imshow('mask-gris', mask_gris)
    #cv2.imshow('mask_bleu', mask_bleu)
    #cv2.imshow('mask_blanc', thresholded_image)
    #cv2.imshow('mask_vert', mask_vert)
    #cv2.imshow('mask_jaune', mask_jaune)
    #cv2.imshow('mask_blanc', gray_image)
    cv2.imshow('noir', image_noire)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_noire

def detect_and_draw(image, mask, label,carte_virtuelle,couleur):
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c = max(elements, key=cv2.contourArea)
        ((x, y), rayon) = cv2.minEnclosingCircle(c)
        carte_virtuelle=draw_circles(carte_virtuelle, x, y,couleur)
        print("coordonné", x,y,label)
        if rayon > 2:
            cv2.circle(mask, (int(x), int(y)), int(rayon), (0, 255, 0), 2)
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 255), 10)
            cv2.line(image, (int(x), int(y)), (int(x) + 150, int(y)), (0, 255, 255), 2)
            cv2.putText(image, label, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    return carte_virtuelle

def custom_white_balance(image, reference_white):

    image_float = image.astype(float)

    scaling_factors = reference_white / image_float.mean(axis=(0, 1))

    balanced_image = image_float * scaling_factors
    balanced_image = np.clip(balanced_image, 0, 255)

    balanced_image = balanced_image.astype(np.uint8)

    return balanced_image


def calibrer_blancs(image):
    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    reference_white = np.array([230, 230, 230])  # Reference white point (for example, pure white)
    image_calib = custom_white_balance(image_rgb, reference_white)

    return cv2.cvtColor(image_calib, cv2.COLOR_RGB2BGR)

def draw_circles(image,x,y,couleur):
    """
    Dessine des cercles sur une image.

    Args:
        image (numpy.ndarray): L'image sur laquelle dessiner les cercles.
        cercles (list): Une liste de tuples contenant les positions et rayons des cercles à dessiner.
                        Chaque tuple est de la forme (x, y, rayon).
        couleur (tuple, optional): La couleur des cercles au format (B, G, R). Par défaut, blanc.

    Returns:
        numpy.ndarray: L'image avec les cercles dessinés.
    """


    cv2.circle(image, (int(x), int(y)), 40, couleur, -1)  # -1 pour remplir le cercle

    return image
def draw_circles2(image,x,y,couleur):
    """
    Dessine des cercles sur une image.

    Args:
        image (numpy.ndarray): L'image sur laquelle dessiner les cercles.
        cercles (list): Une liste de tuples contenant les positions et rayons des cercles à dessiner.
                        Chaque tuple est de la forme (x, y, rayon).
        couleur (tuple, optional): La couleur des cercles au format (B, G, R). Par défaut, blanc.

    Returns:
        numpy.ndarray: L'image avec les cercles dessinés.
    """


    cv2.circle(image, (int(x), int(y)), 5, couleur, -1)  # -1 pour remplir le cercle

    return image
def get_rgb_color_high(image, y, x):
    b, g, r = image[y, x]
    return (r+15, g+15, b+15)

def get_rgb_color_low(image, y, x):
    b, g, r = image[y, x]
    return (r-15, g-15, b-15)


def create_2d_array(rows, cols):
    return [[0] * cols for _ in range(rows)]

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


# Chemin vers votre image contenant le marqueur ArUco
chemin_image = "..\\img\\carte3.jpg"
chemin_image1 = "..\\img\\carte4.jpg"
chemin_image2 = "..\\img\\carte5.jpg"
chemin_image3 = "..\\img\\carte9.jpg"






# Dimensions de l'image
largeur = 1680
hauteur = 945






# Appeler la fonction pour trouver l'ID du marqueur ArUco
ids_marqueurs, coin_H_G, coin_H_D, coin_B_D, coin_B_G = trouver_id_aruco(chemin_image)
image1 = cv2.imread(chemin_image)
pts1 = np.float32([coin_H_G, coin_H_D, coin_B_D, coin_B_G])
pts2 = np.float32([[0, 0], [largeur, 0], [largeur, hauteur], [0, hauteur]])
M = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(image1, M, (largeur, hauteur))

jaune_high = get_rgb_color_high(warped, 145, 932)
jaune_low = get_rgb_color_low(warped, 145, 932)

bleu_ciel_high = get_rgb_color_high(warped, 195, 932)
bleu_ciel_low = get_rgb_color_low(warped, 195, 932)

vert_high = get_rgb_color_high(warped, 245, 932)
vert_low = get_rgb_color_low(warped, 245, 932)

rose_high = get_rgb_color_high(warped, 295, 932)
rose_low = get_rgb_color_low(warped, 295, 932)

rouge_high = get_rgb_color_high(warped, 345, 932)
rouge_low = get_rgb_color_low(warped, 345, 932)

bleu_marine_high = get_rgb_color_high(warped, 395, 932)
bleu_marine_low = get_rgb_color_low(warped, 395, 932)

noir_high = get_rgb_color_high(warped, 445, 932)
noir_low = get_rgb_color_low(warped, 445, 932)


# Appeler la fonction de calibrage des blancs avec le chemin spécifié
#calibrer_blancs(warped)
cv2.imshow('Image calibrée', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Afficher les IDs des marqueurs détectés

if ids_marqueurs is not None:
  print("IDs des marqueurs détectés :", ids_marqueurs)

# Paramètres HSV pour les différentes couleurs
#lo_rouge = np.array([rouge_low])
#hi_rouge = np.array([rouge_high])
lo_rouge = np.array([0,100,100])
hi_rouge = np.array([10,255,255])
lo_gris = np.array([70,70,70])
hi_gris = np.array([150,150,150])
#lo_bleu = np.array([bleu_ciel_low])
#hi_bleu = np.array([bleu_ciel_high])
#lo_vert = np.array([vert_low])
#hi_vert = np.array([vert_high])
#lo_jaune = np.array([jaune_low])
#hi_jaune = np.array([jaune_high])
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
carte_virtuelle=detect_cubes_in_image(warped, lo_rouge, hi_rouge, lo_gris, hi_gris, lo_bleu, hi_bleu, lo_vert, hi_vert, lo_jaune,
                      hi_jaune, lo_blanc, hi_blanc)

#ids_marqueurs, coin_H_G, coin_H_D, coin_B_D, coin_B_G = trouver_id_aruco2(warped)



# Créer le tableau 2D avec les lignes et les colonnes spécifiées
maze = create_2d_array(largeur, hauteur)

# Vérifier si l'image est chargée avec succès
if carte_virtuelle is not None:
    # Coordonnées du pixel à lire (par exemple, pixel à la position (100, 100))
    for i in range(1600):
        for j in range(900):
            x, y = i, j
            # Lire les valeurs des canaux de couleur du pixel
            b, g, r = carte_virtuelle[y, x]
            if r!=0 or b!=0 or g!=0:
                maze[i][j] = 1





start = (100, 80)
end = (1200, 700)

path = astar(maze, start, end)
for x,y in path :
    carte_virtuelle=draw_circles2(carte_virtuelle,x,y,(0,255,255))


print(path)
cv2.imshow('Image final!!!!!!!!!!!!', carte_virtuelle)

cv2.waitKey(0)
cv2.destroyAllWindows()