import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Définir les sommets du cube centré en 0 avec une taille de 100
half_size = 50
vertices = np.array([[-half_size, -half_size, -half_size],
                     [half_size, -half_size, -half_size],
                     [half_size, half_size, -half_size],
                     [-half_size, half_size, -half_size],
                     [-half_size, -half_size, half_size],
                     [half_size, -half_size, half_size],
                     [half_size, half_size, half_size],
                     [-half_size, half_size, half_size]])

# Définir les faces du cube en termes de sommets
faces = [[vertices[j] for j in [0, 1, 2, 3]],
         [vertices[j] for j in [4, 5, 6, 7]],
         [vertices[j] for j in [0, 3, 7, 4]],
         [vertices[j] for j in [1, 2, 6, 5]],
         [vertices[j] for j in [0, 1, 5, 4]],
         [vertices[j] for j in [2, 3, 7, 6]]]


# Créer la figure et l'axe 3D avec une taille d'image de 250
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111, projection='3d')

# Ajouter les faces à la figure
ax.add_collection3d(Poly3DCollection(faces, 
                                    facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

# Définir les limites des axes
ax.set_xlim([-half_size, half_size])
ax.set_ylim([-half_size, half_size])
ax.set_zlim([-half_size, half_size])

# Ajouter les étiquettes des axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Afficher le cube
plt.show()
