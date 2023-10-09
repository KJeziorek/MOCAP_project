import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np


def draw_cube(_img, P):
    """Function draw a unit qube on an image"""
    
    img = _img.copy()
    cube = [[0,0,0], [0.35,0,0], [0, 0.35,0], [0.35,0.35,0], [0,0,0.35], [0.35,0,0.35], [0,0.35,0.35], [0.35,0.35,0.35]]
    cube_lines = lambda cube: [(cube[0],cube[1]), (cube[0],cube[2]),(cube[3],cube[1]), (cube[3],cube[2]),(cube[4],cube[5]), (cube[4],cube[6]),(cube[7],cube[5]), (cube[7],cube[6]),(cube[0],cube[4]),(cube[1],cube[5]),(cube[2],cube[6]),(cube[3],cube[7])]

    # Translate the cube cooridinates to the iamge cooridnates
    img_cube = []
    for vec in cube:
        # Project the 3D point to 2D image
        proj = P @ np.append(vec, 1)
        # Normalize the vector
        proj /= proj[-1]
        img_cube.append(proj[:-1])

    img_lines = cube_lines(img_cube)

    # Read image and draw lines
    for line in img_lines:
        line = np.squeeze(line)
        img = cv2.line(img, line[0].astype(int), line[1].astype(int), (255,0,0), 1)

    return img

def plot_cameras_in_space(cameras, patter):
    """ Plot cameras with marker on a 3D plot"""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for id, cam in enumerate(cameras):
        r_mat = cam["R"]
        t_vec = cam["t"]
        coord = -np.matrix(r_mat).T @ np.matrix(t_vec)

        ax.scatter3D(coord[0], coord[1], coord[2], label=f'cam: {id+1}')
    ax.set_aspect('auto')

    ax.set_aspect('auto')
    # Draw marker
    p = np.array(list(patter.values()))
    ax.scatter(p[:,0], p[:,1], p[:,2],marker='*',label='Markers')
    return fig

def draw_lines(F, img1, img2, sorted1, sorted2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    r, c, _ = img2.shape

    pts1 = np.array([ (x, y) for _, x, y in sorted1],dtype='int32')
    pts2 = np.array([ (x, y) for _, x, y in sorted2],dtype='int32')

    for point1, point2 in zip(pts1, pts2):
        line1 = F @ np.append(point1, 1)
        line2 = np.transpose(F) @ np.append(point2, 1)

        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -line1[2] / line1[1]])
        x1, y1 = map(int, [c, -(line1[2] + line1[0] * c) / line1[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, point2, 1, color, 3)

        x0, y0 = map(int, [0, -line2[2] / line2[1]])
        x1, y1 = map(int, [c, -(line2[2] + line2[0] * c) / line2[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, point1, 1, color, 3)
        Hori = np.concatenate((img1, img2), axis=1)
        # cv2.imshow("Line", Hori)
        # cv2.waitKey(50)

    return img1, img2, Hori
