import numpy as np

def circle(balloon_param):
    K, x0, y0, R = balloon_param
    theta = np.linspace(0 ,2*np.pi, K)
    x = x0 + R*np.cos(theta)
    y = y0 + R*np.sin(theta)
    return (x, y)

def sphere(balloon_param):
    K, x0, y0, z0, rho = balloon_param
    size_grid = int(np.sqrt(K))
    x = np.linspace(0, K - 1, K)
    y = np.linspace(0, K - 1, K)
    z = np.zeros(K)
    x, y, z = np.meshgrid(x, y, z)

    dt = np.linspace(0, 2*np.pi, K)
    dr = np.linspace(0, 2*np.pi, K)
    for u in range(size_grid):
        for v in range(size_grid):
            x[u, v] = x0 + rho*np.cos(dt)*np.cos(dr)
            y[u, v] = y0 + rho*np.sin(dt)*np.cos(dr)
            z[u, v] = z0 + rho*np.sin(dr)
    return (x, y, z)

def check_balloon(name):
    if name in BALLOON_NAME:
        return True
    return False

BALLOON_NAME = {
    'circle' : circle,
    'sphere' : sphere
}    