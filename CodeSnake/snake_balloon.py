import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.interpolate import CubicSpline
import cv2
import triangle as tr

from balloon import *


def update_A(K, alpha, beta, dt):
    D2 = np.diag(-2*np.ones((1,K))[0],0) + np.diag(np.ones((1,K-1))[0],1) + np.diag(np.ones((1,K-1))[0],-1)
    D2[0][-1] = D2[-1][0] = 1

    D4 = np.diag(6*np.ones((1,K))[0],0) + np.diag(-4*np.ones((1,K-1))[0],1) + np.diag(-4*np.ones((1,K-1))[0],-1) + np.diag(np.ones((1,K-2))[0],2) + np.diag(np.ones((1,K-2))[0],-2)
    D4[0][-1] = D4[-1][0] = -4
    D4[0][-2] = D4[1][-1] = D4[-1][1] = D4[-2][0] = 1

    D = alpha*D2 - beta*D4
    A = np.linalg.inv(np.eye(K,K) - dt*D)
    return A


def snake_balloon_2D(I_opened,I, balloon_param, param):
    IMAGES = [I_opened]
    scale_x, scale_y = len(I)/10, len(I[0])/10
    print(scale_x, scale_y)

    iteration = param['iteration']
    alpha = param["alpha"]
    beta = param["beta"]
    gamma = param["gamma"]
    kappa = param["kappa"]
    dt = param["dt"]
    K = balloon_param[1]

    balloon_name = balloon_param.pop(0)
    if not check_balloon(balloon_name):
        return "Ce balloon n'existe pas"
    balloon = BALLOON_NAME[balloon_name]

    x, y = balloon(balloon_param)

    # Définir les points du triangle
    pt1 = (100, 200)  # Premier sommet
    pt2 = (245, 295)  # Deuxième sommet
    pt3 = (250, 100)  # Troisième sommet

    #x, y=tr.triangle(pt1,pt2,pt3,K)
    
    grad_I_x, grad_I_y = np.gradient(I_opened)
    norm_grad = grad_I_x**2 + grad_I_y**2
    norm_grad = cv2.GaussianBlur(norm_grad, (11,11), 0)
    grad_x, grad_y = np.gradient(norm_grad)
    IMAGES.append(norm_grad)

    New_K=K

    x = np.transpose(x)
    y = np.transpose(y)

    CONTOUR_IMAGE = []

    for i in range(iteration):
        A = update_A(New_K, alpha, beta, dt)
        ti_more_x = np.roll(x, - 1)
        ti_minus_x = np.roll(x, 1)
        ti_more_y = np.roll(y, - 1)
        ti_minus_y = np.roll(y, 1)  
        ti_x = ti_more_x - ti_minus_x
        ti_y = ti_more_y - ti_minus_y 

        norm_ti = np.sqrt(ti_x**2 + ti_y**2)

        n_x = -ti_y/norm_ti
        n_y = ti_x/norm_ti

        xi = np.dot(A, x + dt*gamma*(grad_x[x.astype(int),y.astype(int)] + kappa*n_x))
        yi = np.dot(A, y + dt*gamma*(grad_y[x.astype(int),y.astype(int)] + kappa*n_y))

        x = xi
        y = yi

        c = list()
        cc = np.zeros((K, 1, 2))
        cc[:,0,0] = y
        cc[:,0,1] = x 
        c.append(cc.astype(int))  

        #Calcul de la distance des points à chaque itération
        dist_l=np.sqrt((x-ti_more_x)**2 + (y - ti_more_y)**2)
        #Liste nouveau point d'interpolation 
        point_interpol_x=[]
        point_interpol_y=[]  

        for i in range(len(dist_l)):
            if dist_l[i]>10:
                x_cub = [x[i-2], x[i-1], x[i], x[i+1]]
                y_cub = [y[i-2], y[i-1], y[i], y[i+1]]

                # Créer l'interpolation cubique
                cs = CubicSpline(x_cub, y_cub)
                # Calculer l'ordonnée du nouveau point
                x_new=(x[i]+x[i+1])/2
                y_new = cs(x_new)
                point_interpol_x.append([x_new,i])
                point_interpol_y.append([y_new,i])

            elif dist_l[i]<5:
                np.delete(x,i)
                np.delete(y,i)

        for i in range(len(point_interpol_x)):
            x.insert(point_interpol_x[1],point_interpol_x[0])
            x.insert(point_interpol_y[1],point_interpol_y[0])


        New_K=len(x)

        
        if i % 100 == 0:
            I_c = cv2.drawContours(image=cv2.cvtColor(I, cv2.COLOR_GRAY2BGR), contours=c, contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            I_c = cv2.putText(I_c, f"Iteration: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            I_c = cv2.putText(I_c, f"alpha: {alpha}", (0, 9*int(scale_x)), cv2.FONT_HERSHEY_SIMPLEX, 4/np.sqrt(scale_x*scale_y), (255, 0, 0), 1, cv2.LINE_AA)

            CONTOUR_IMAGE.append(I_c)

    return IMAGES, CONTOUR_IMAGE