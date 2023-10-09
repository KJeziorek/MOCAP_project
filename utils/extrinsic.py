import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np

def compute_homography(img_points, obj_points):
    A = []
    for i in range(0, len(img_points)):
        u, v = img_points[i][0], img_points[i][1]
        x, y = obj_points[i][0], obj_points[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def find_camera_position(img_points, obj_points, K):
    """Find camera position by solving PnP problem for a coplanar marker"""
    img_points = np.float32(img_points)
    obj_points = np.float32(obj_points)

    # Asume no camera distortion
    dist_coeffs = np.zeros((4,1))
    
    success, r_vec, t_vec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    r_vec, t_vec = cv2.solvePnPRefineVVS(obj_points, img_points, K,  dist_coeffs, r_vec, t_vec)

    r_mat = cv2.Rodrigues(r_vec)[0]

    return r_mat, t_vec

def calculate_projection_matrix(r_mat, t_vec, K):
    return K @ np.hstack([r_mat,t_vec])

def triangulate_points(P1, P2, p1, p2):
    point = cv2.triangulatePoints(P1, P2, p1, p2)
    point /= point[-1]

    return point[:-1]

def n_view_traingulation(P_vec, img_points):
    """
    Created this function based on webpage: 
    https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/
    P_vec - vector of P arrays for each camera
    img_points - corestornding point in the image
    """
    create_row = lambda u, v, P : np.vstack(( u*P[2,:]-P[0,:],
                                              v*P[2,:]-P[1,:]))

    A = np.vstack([create_row(u,v,P) for (u, v), P  in zip(img_points, P_vec)])
    
    # Solve the A*X = 0 using SVD
    u, s, vh = np.linalg.svd(A)
    X = vh[-1,:]

    return X/X[-1]

def computeFundamentalMatrix(left_projection, right_projection):
    # Na podstawie https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-16-fundamental-essential.pdf (str.9 pdf)
    # Macierz F wyznacza linie epipolarną na obrazie right na podstawie punktu z obrazu left.
    R_left, t_left = left_projection[:, 0:3], left_projection[:, 3]
    R_right, t_right = right_projection[:, 0:3], right_projection[:, 3]

    b = np.linalg.inv(R_right) @ t_right - np.linalg.inv(R_left) @ t_left
    Sb = np.array([[0, -b[2], b[1]],
                   [b[2], 0, -b[0]],
                   [-b[1], b[0], 0]])

    F = np.transpose(np.linalg.inv(R_left)) @ Sb @ np.linalg.inv(R_right)
    return np.transpose(F)