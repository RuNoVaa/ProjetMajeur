from snake_balloon import *
from display import *
from polygon_3D import *
"""
balloon_param = [name -> str, pos_x -> float, pos_y -> float, K -> int, ...]
"""

if __name__ == "__main__":
    I = cv2.imread("./CodeSnake/Images/im9.png", 0)
    I = cv2.normalize(I, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    I_3D = create_cube((40, 40, 40), 10, (20, 20, 20))
    balloon_param = ["circle", 700, len(I)/2 + 5, len(I[0])/2 + 5, 5]
    balloon_param_3D = ["sphere", 100, 0, 0, 0, 10]
    param = {
        "alpha": 0.15,
        "beta": 0.001,
        "gamma": 50,
        "kappa": -0.006,
        "sigma": 7,
        "dt": 0.1,
        "iteration" : 3001
    }
    IMAGES, CONTOUR_IMAGE = snake_balloon_2D(I, balloon_param, param)
    # snake_balloon_3D(I_3D, balloon_param_3D, param)
    # plot_cube(I_3D)
    display(CONTOUR_IMAGE, "doigt1")