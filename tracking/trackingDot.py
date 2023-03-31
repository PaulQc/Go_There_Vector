# Code pour la détection de la position de Vector à l'aide des pastilles de couleur
# Présentement codé pour la configuration:
# - Une pastille rouge à gauche et une pastille bleu à droite
# - Séparé par environ 30 mm
# - Ajoute une segmentation de Vector par une technique ORB pour limiter
#   les problèmes de mauvaise identification des pastilles dû à d'autres objets
#
# Le code essai aussi de contourner les erreur que OpenCV donne assez fréquemment dans l'analyse
# des images
#
# Adaptation de la méthode développé par https://github.com/automaticdai/rpi-object-detection
#
# Paul Grenier 2023-01-17
#
import cv2
import numpy as np

# Calibration des images
cal_mm = 1.05  # mm par pixel
# Valeur de h, s et v pour les pastilles de couleur
dot_color = ('blue', 'red')
dot_color_hsv = {dot_color[0]: {'hsv_min': [105, 100, 180], 'hsv_max': [125, 255, 255]},
                 dot_color[1]: {'hsv_min': [140, 50, 50], 'hsv_max': [180, 255, 255]}
                 }
dot_max_separation = 36/cal_mm  # en pixel

# Prépare la détection ORB et l'analyse k-means pour mieux cibler l'endroit où est Vector
# Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
orb = cv2.ORB_create(1000, 1.5)
# Load our image template, this is our reference image
image_ref = cv2.imread('Vector_imgcrop.jpg', 0)
# Detect keypoints of reference image
(kp2, des2) = orb.detectAndCompute(image_ref, None)
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
nb_cluster = 5


def find_vector_area(frame):
    """
    Méthode qui détermine plus précisément la position de Vector dans le frame à l'aide
    de la technique ORB. Retourne les coordonnés d'un rectangle qui devrait l'englober
    :param frame:
    :return: top_left_x, top_left_y, bottom_right_x, bottom_right_y
    """
    image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)
    # Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Do matching
    matches = bf.match(des1, des2)
    # Sort the matches based on distance.  Least distance is better
    matches = sorted(matches, key=lambda val: val.distance)
    nb_matches = len(matches)
    list_pt1 = []
    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        list_pt1.append((x1, y1))

    z = np.float32(np.array(list_pt1, dtype=float))  # Définir en float32 semble essentiel !
    width = np.max([np.max(np.nanstd(z, axis=0)), 100])
    # Utilise k-means pour mieux cibler l'endroit où est Vector
    compactness, labels, centers = cv2.kmeans(z, nb_cluster, None, criteria, 40, flags)
    nb_in_group = [np.sum(labels == i) for i in range(nb_cluster)]
    no_group_max = np.where(nb_in_group == np.max(nb_in_group))[0][0]
    centre = centers[no_group_max, :]  # Suppose que Vector est là où nb_in_group est maximum
    # Define ROI Box Dimensions
    top_left_x = int(centre[0] - width / 2)
    if top_left_x <= 0:
        top_left_x = 1
    top_left_y = int(centre[1] - width / 2)
    if top_left_y <= 0:
        top_left_y = 1
    bottom_right_x = int(centre[0] + width / 2)
    if bottom_right_x >= 640:
        bottom_right_x = 639
    bottom_right_y = int(centre[1] + width / 2)
    if bottom_right_y >= 480:
        bottom_right_y = 479
    #
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


class TrackingDot:
    """
    Class pour les calculs de localisation et de trajectoire de Vector dans son espace de jeu
    """
    def __init__(self):
        self.id = []
        self.position = (0, 0)
        self.orientation = (0, 0)
        self.dot_position = {dot_color[0]: (0, 0), dot_color[1]: (0, 0)}
        self.rotation = 0.0
        self.distance = 0.0
        # Série de Flag pour cerner l'origine de l'erreur
        self.ok_flag_position = True
        self.ok_flag_separation = True
        self.ok_flag_dot_position = {dot_color[0]: True, dot_color[1]: True}
        self.ok_flag_cv2 = True  # Si la méthode de OpenCV a exécuté sans erreur

    def define_path(self, frame, target_point):
        """
        Méthode qui détermine les déplacements que Vector doit exécuter pour ce rendre
        au point désiré (target_point) définie comme à un pixel du frame (px, py)
        :return True si l'estimé de la trajectoire semble ok
        """
        # Détermine la position des pastilles de couleur
        # Premièrement, précise où est Vector à l'aide de la technique ORB
        x_top_l, y_top_l, x_bot_r, y_bot_r = find_vector_area(frame)
        # Détermine la position des pastilles dans l'image réduite
        frame_cropped = frame[y_top_l:y_bot_r, x_top_l:x_bot_r]
        for color in dot_color:
            self.ok_flag_dot_position[color] = False
            self.dot_position[color] = (0, 0)
            self.ok_flag_cv2 = False
            self.find_dot_position(frame_cropped, color)
            if not self.ok_flag_dot_position[color]:
                return False
        # Redefinie la position des pastilles p/r au frame complet 640x480
        for color in dot_color:
            dot_pos = np.array(self.dot_position[color]) + np.array([x_top_l, y_top_l])
            self.dot_position[color] = tuple(dot_pos)
        # Vérifie si la séparation entre les pastilles est anormalement élevée
        separation = np.sqrt(np.sum((np.array(self.dot_position[dot_color[0]]) -
                                     np.array(self.dot_position[dot_color[1]])) ** 2))
        if separation > dot_max_separation:
            self.ok_flag_separation = False
            return False
        # Détermine la trajectoire pour se rendre à la position désirée
        # Utilise la notation de mon calcul manuscrit
        b = np.array(self.dot_position[dot_color[0]], dtype=float)
        r = np.array(self.dot_position[dot_color[1]], dtype=float)
        pd = np.array(target_point, dtype=float)
        #
        vp = r + (b - r) / 2
        self.position = (int(vp[0]), int(vp[1]))  # Mis en tuple (px_X, px_Y)
        self.ok_flag_position = True
        # Translation des coordonnées à ce point
        r_t = r - vp
        b_t = b - vp
        pd_t = pd - vp
        # Calcul l'orientation/direction de Vecteur
        vd = np.array([0.0, 0.0])
        # Vérifie les cas où vd serait parallèle à une des 2 axe x ou y
        # avant de faire le calcul pour un cas quelconques (pour éviter une division par 0)
        if np.abs((r_t[0] - b_t[0])) < 0.5:
            vd[0] = -1 * np.sign((r_t[1] - b_t[1]))
            vd[1] = 0
        else:
            if np.abs((r_t[1] - b_t[1])) < 0.5:
                vd[0] = 0
                vd[1] = 1 * np.sign((r_t[0] - b_t[0]))
            else:
                vd[0] = np.sqrt(1 / (1 + (r_t[0] - b_t[0]) ** 2 / (r_t[1] - b_t[1]) ** 2))
                if r_t[1] >= 0:
                    vd[0] = -1 * vd[0]
                vd[1] = -1 * vd[0] * (r_t[0] - b_t[0]) / (r_t[1] - b_t[1])
        self.orientation = vd
        # Rotation pour amener les coordonnées selon l'orientation de Vector
        tetha_coor = np.arccos(vd[0])
        if vd[1] < 0:
            tetha_coor = 2 * np.pi - tetha_coor
        rot = np.array([[np.cos(tetha_coor), np.sin(tetha_coor)], [-1 * np.sin(tetha_coor), np.cos(tetha_coor)]])
        pd_t_r = np.matmul(rot, pd_t)
        # Calcul alors l'angle que doit tourner Vector pour se rendre à la position désiré
        distance = np.sqrt(np.sum(pd_t_r ** 2))
        tetha_v = np.arccos(pd_t_r[0] / distance)
        if pd_t_r[1] > 0:
            tetha_v = -1 * tetha_v  # rotation clockwise dans ce cas
        # Calibre les valeurs et les compiles dans le dictionnaire
        self.rotation = tetha_v * 180 / np.pi  # Angle en degré
        self.distance = distance * cal_mm  # Distance à parcourir en mm
        return True

    def find_dot_position(self, frame, color):
        """
        Méthode qui détermine la position de la pastille dans ce frame
        """
        # Initialize comme si problème jusqu'à preuve du contraire
        self.ok_flag_dot_position[color] = False
        self.ok_flag_cv2 = False  # i.e probleme du a OpenCV
        try:
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except:
            return
        try:
            thresh = cv2.inRange(frame_hsv, np.array(dot_color_hsv[color]['hsv_min']),
                                 np.array(dot_color_hsv[color]['hsv_max']))
        except:
            return
        # find contours in the threshold image
        try:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            return
        # finding contour with maximum area and store it as best_cnt
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt
        self.ok_flag_cv2 = True  # i.e pas un probleme avec OpenCV rendu ici
        # Extract centroids of best_cnt
        if max_area != 0:
            M = cv2.moments(best_cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            self.dot_position[color] = (cx, cy)
            self.ok_flag_dot_position[color] = True
        return
