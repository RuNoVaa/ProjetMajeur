import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_cube(size_I, size_cube, pos):
    x0, y0, z0 = pos
    # Create an empty array of zeros with the specified size
    I = np.zeros(size_I)
    
    # Calculate the half size of the cube
    half_size_cube = size_cube // 2
    
    # Determine the bounds of the cube
    x_start = max(x0 - half_size_cube, 0)
    x_end = min(x0 + half_size_cube, size_I[0])
    y_start = max(y0 - half_size_cube, 0)
    y_end = min(y0 + half_size_cube, size_I[1])
    z_start = max(z0 - half_size_cube, 0)
    z_end = min(z0 + half_size_cube, size_I[2])
    
    # Fill the region corresponding to the cube with ones
    I[x_start:x_end, y_start:y_end, z_start:z_end] = 1
    
    return I

