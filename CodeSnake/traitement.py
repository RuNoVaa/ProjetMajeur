import cv2
from matplotlib import pyplot as plt

I = cv2.imread("./Images/breast-implant.tif", 0)
I = cv2.normalize(I, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# Définir l'élément structurant (un noyau carré de taille 5x5 dans cet exemple)
S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# Appliquer l'opération d'ouverture
E=cv2.morphologyEx(I, cv2.MORPH_OPEN, S)
E=cv2.blur(E, (5,5))
plt.imshow(E,cmap='gray')
plt.show()