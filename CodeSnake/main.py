from snake_balloon import *
from display import *
"""
balloon_param = [name -> str, pos_x -> float, pos_y -> float, K -> int, ...]
"""

if __name__ == "__main__":
    I = cv2.imread("./Images/im1.png", 0)
    I = cv2.normalize(I, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    balloon_param = ["circle", 100, len(I)/2, len(I[0])/2, 10]
    # balloon_param2 = ["sphere", 100, 0, 0, 0, 10]
    param = {
        "alpha": 0.001,
        "beta": 0.0001,
        "gamma": 50,
        "kappa": -0.005,
        "sigma": 7,
        "dt": 0.1,
        "iteration" : 3001
    }
    IMAGES, CONTOUR_IMAGE = snake_balloon_2D(I, balloon_param, param)
    # snake_balloon_3D(I, balloon_param2, param)
    display(CONTOUR_IMAGE)