import numpy as np



def circle(balloon_param):
    K, x0, y0, R = balloon_param
    theta = np.linspace(0 ,2*np.pi, K)
    x = x0 + R*np.cos(theta)
    y = y0 + R*np.sin(theta)
    return (x, y)

def check_balloon(name):
    if name in BALLOON_NAME:
        return True
    return False

BALLOON_NAME = {
    'circle' : circle
}    