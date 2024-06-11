import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diff_u(x, y, z):
    x_u_more = np.roll(x, -1, axis = 0)
    x_u_minus = np.roll(x, 1, axis = 0)
    y_u_more = np.roll(y, -1, axis = 0)
    y_u_minus = np.roll(y, 1, axis = 0)
    z_u_more = np.roll(z, -1, axis = 0)
    z_u_minus = np.roll(z, 1, axis = 0)

    t_u_x = x_u_more - x_u_minus
    t_u_y = y_u_more - y_u_minus
    t_u_z = z_u_more - z_u_minus

    return t_u_x, t_u_y, t_u_z

def diff_v(x, y, z):
    x_v_more = np.roll(x, -1, axis = 1)
    x_v_minus = np.roll(x, 1, axis = 1)
    y_v_more = np.roll(y, -1, axis = 1)
    y_v_minus = np.roll(y, 1, axis = 1)
    z_v_more = np.roll(z, -1, axis = 1)
    z_v_minus = np.roll(z, 1, axis = 1)
    
    t_v_x = x_v_more - x_v_minus
    t_v_y = y_v_more - y_v_minus
    t_v_z = z_v_more - z_v_minus

    return t_v_x, t_v_y, t_v_z

def normal(t_u, t_v):
    n_x = t_u[1]*t_v[2] - t_u[2]*t_v[1]
    n_y = t_u[2]*t_v[0] - t_u[0]*t_v[2]
    n_z = t_u[0]*t_v[1] - t_u[1]*t_v[0]

    norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)

    n_x /= norm
    n_y /= norm
    n_z /= norm

    return n_x, n_y, n_z












