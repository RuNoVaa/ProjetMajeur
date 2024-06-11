from snake_balloon import *
from display import *
"""
balloon_param = [name -> str, pos_x -> float, pos_y -> float, K -> int, ...]
"""

if __name__ == "__main__":
    I = cv2.imread("./CodeSnake/Images/im1.png", 0)
    I = cv2.normalize(I, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Définir l'élément structurant ici pour crapbulsar-optical.tif
    #S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Appliquer l'opération d'ouverture
    E=cv2.morphologyEx(I, cv2.MORPH_OPEN, S)

    balloon_param = ["circle", 100, len(I)/2, len(I[0])/2, 10]
    param = {
        #Crabpulsar
        #Appliquer l'opération d'ouverture, nécessaire pour le traitement d'image

        #E=cv2.erode(I, S, 1)
        #"alpha": 4.5,
        #"beta": 0.5,
        #"gamma": 290,
        #"kappa": -0.00048,
        #["circle", 500, len(I)/2, len(I[0])/2+20, 10]
        #Alpha et Beta pas trop élévé puisque qu'on perd peu à peu la forme du cercle. Mais pas trop élévé sinon on a des pointes qui se créent.
        #Gamma adapté pour que le gradient n'affecte pas dans la cellule entre le blanc et le gris. On adapte ensuite kappa pour le gonflement.

        #Paramètre pour illusion-optique
        #"alpha": 0.000000000000001,
        #"beta": 1000,
        #"gamma": 250,
        #"kappa": 0.001,
        #Un alpha très faible pour l'élasticité, beta élévé pour conserver le forme, gamme élévé pour attirer sur le fort gradient sur les contours, et un kappa adpaté pour gonfler

        #Paramètres breast-implant 
        #"alpha": 1,
        #"beta": 0.00000000000001,
        #"gamma": 2000,
        #"kappa": -0.0003,
        #Un beta très faible pour que le cercle puisse prendre un creu, un alpha peu élevé pour laisser l'élastique, un gamma élevé car on doit s'arrêter sur la zone de fort gradient et kappa adapté

        "alpha": 1,
        "beta": 0.00000000000001,
        "gamma": 2000,
        "kappa": -0.004,
        "dt": 0.1,
        "iteration" : 2401
    }
    IMAGES, CONTOUR_IMAGE = snake_balloon_2D(E,I, balloon_param, param)
    display(CONTOUR_IMAGE)