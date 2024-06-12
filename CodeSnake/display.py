import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.imshow(frame)
    ax.axis('off')

def display(I):
    ani = animation.FuncAnimation(fig, update, frames=I, repeat=False)
    plt.show()

def plot_cube(I):
    # Create a figure and a 3D axis
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the indices of the points in the array that have value 1
    x, y, z = np.indices(I.shape)
    cube = (I == 1)
    
    # Plot the cube
    ax.voxels(cube, facecolors='blue', edgecolors='gray', alpha=0.5)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
