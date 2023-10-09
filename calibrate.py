from utils.intrinsic import calibrate_camera
from utils.markers import markers_to_points, find_corresponding_points
from utils.extrinsic import calculate_projection_matrix, compute_homography, find_camera_position, computeFundamentalMatrix
from utils.errors import calculate_error, calculate_error_pattern
from utils.visual import draw_cube, plot_cameras_in_space, draw_lines

from typing import List, Tuple
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

num_of_cameras = 4
cameras = []
undistort = False

patterns = {0: [0, 0, 0], 1: [0.175, 0, 0], 2: [0.35, 0, 0], 3: [0.35, 0.35, 0], 4: [0, 0.35, 0], 5: [0, 0.175, 0]}

if __name__ == "__main__":
    for i in range(num_of_cameras):

        os.makedirs(f'output/calibration/camera_{i+1}', exist_ok = True)

        camera = {}
        ####### Intrinsic camera calibration #######

        dir_intr = f"data/inter_calibration/camera_{i+1}"
        K, dist = calibrate_camera(dir_intr, vis=False)

        camera["K"] = K
        camera["dist"] = dist

        ####### Extrinsic camera calibration #######

        dir_extr = f"data/exter_calibration/camera_{i+1}.jpg"
        camera["img"] = dir_extr
        img = cv2.imread(dir_extr)

        cv2.imwrite(f'output/calibration/camera_{i+1}/original.jpg', img)

        # Undistort?
        if undistort:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, K, dist, None, newcameramtx)
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]

        # Find markers on a image
        markers, preprocessed_img = markers_to_points(img)

        cv2.imwrite(f'output/calibration/camera_{i+1}/preprocessed.jpg', preprocessed_img)

        # Sort the markers
        sorted_markers = find_corresponding_points(markers)
        camera["markers"] = sorted_markers

        # Find the Homeography
        img_points = np.array([ (x, y) for _, x, y in sorted_markers],dtype='double')
        obj_points = np.array([ patterns[id] for id, _, _ in sorted_markers],dtype='double')
        
        H = compute_homography(img_points, obj_points)
        camera["H"] = H 

        r_mat, t_vec = find_camera_position(img_points, obj_points, K)
        camera["R"] = r_mat
        camera["t"] = t_vec

        # Compute pose from homeography (compute P)
        P = calculate_projection_matrix(r_mat, t_vec, K)
        camera["P"] = P

        # Draw cube
        cube_image = draw_cube(img, P)

        cv2.imwrite(f'output/calibration/camera_{i+1}/cube.jpg', cube_image)
        
        # Calculate re-projection error
        pts, proj, err, sse = calculate_error(img_points, obj_points, P)

        with open(f'output/calibration/camera_{i+1}/error.txt', 'w') as f:
            f.write(f"Sum of squred errors of reproductions is: {sse}\n")
            for _pts, _proj, _err in zip(pts, proj, err):
                f.write(f"point:\t\t{_pts}\nprojection:\t{_proj}\nerror: \t\t{_err}\n\n")

        cameras.append(camera)

        with open(f'output/calibration/camera_{i+1}/camera.txt', 'w') as f:
            for key, values in camera.items():
                f.write(f"{str(key)}:{str(values)}\n\n")

        
    calculate_error_pattern(cameras, patterns)
    space = plot_cameras_in_space(cameras, patterns)

    space.savefig('output/calibration/space.jpg')
    
    for i in range(4):
        for j in range(4):
            if i != j:
                F = computeFundamentalMatrix(cameras[i]["P"], cameras[j]["P"])
                img1, img2, concat = draw_lines(F, cameras[i]["img"], cameras[j]["img"], cameras[i]["markers"], cameras[j]["markers"])
                
                os.makedirs(f'output/calibration/epipolar/camera_{i+1}_camera_{j+1}', exist_ok = True)
                cv2.imwrite(f'output/calibration/epipolar/camera_{i+1}_camera_{j+1}/camera_{i+1}.jpg', img1)
                cv2.imwrite(f'output/calibration/epipolar/camera_{i+1}_camera_{j+1}/camera_{j+1}.jpg', img2)
                cv2.imwrite(f'output/calibration/epipolar/camera_{i+1}_camera_{j+1}/both.jpg', concat)
    
    
    ###### test 1 #######
    print('###### test 1 #######')

    new_cameras = []

    for i, camera in enumerate(cameras):
        
        os.makedirs(f'output/test1/camera_{i+1}', exist_ok = True)

        ####### Extrinsic camera calibration #######

        dir_extr = f"data/test1/camera_{i+1}.jpg"
        camera["img"] = dir_extr
        img = cv2.imread(dir_extr)

        cv2.imwrite(f'output/test1/camera_{i+1}/original.jpg', img)

        # Undistort?
        if undistort:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera["K"], camera["dist"], (w,h), 1, (w,h))
            dst = cv2.undistort(img, camera["K"], camera["dist"], None, newcameramtx)
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]

        # Find markers on a image
        markers, preprocessed_img = markers_to_points(img)

        cv2.imwrite(f'output/test1/camera_{i+1}/preprocessed.jpg', preprocessed_img)

        # Sort the markers
        sorted_markers = find_corresponding_points(markers)
        camera["markers"] = sorted_markers

        # Find the Homeography
        img_points = np.array([ (x, y) for _, x, y in sorted_markers],dtype='double')
        obj_points = np.array([ patterns[id] for id, _, _ in sorted_markers],dtype='double')
        
        new_cameras.append(camera)
    
    calculate_error_pattern(new_cameras, patterns)
    space = plot_cameras_in_space(new_cameras, patterns)

    space.savefig('output/test1/space.jpg')
    
    for i in range(4):
        for j in range(4):
            if i != j:
                F = computeFundamentalMatrix(new_cameras[i]["P"], new_cameras[j]["P"])
                img1, img2, concat = draw_lines(F, new_cameras[i]["img"], new_cameras[j]["img"], new_cameras[i]["markers"], new_cameras[j]["markers"])
                
                os.makedirs(f'output/test1/epipolar/camera_{i+1}_camera_{j+1}', exist_ok = True)
                cv2.imwrite(f'output/test1/epipolar/camera_{i+1}_camera_{j+1}/camera_{i+1}.jpg', img1)
                cv2.imwrite(f'output/test1/epipolar/camera_{i+1}_camera_{j+1}/camera_{j+1}.jpg', img2)
                cv2.imwrite(f'output/test1/epipolar/camera_{i+1}_camera_{j+1}/both.jpg', concat)
    

    ###### test 2 #######
    print('###### test 2 #######')

    new_cameras = []

    for i, camera in enumerate(cameras):
        
        os.makedirs(f'output/test2/camera_{i+1}', exist_ok = True)

        ####### Extrinsic camera calibration #######

        dir_extr = f"data/test2/camera_{i+1}.jpg"
        camera["img"] = dir_extr
        img = cv2.imread(dir_extr)

        cv2.imwrite(f'output/test2/camera_{i+1}/original.jpg', img)

        # Undistort?
        if undistort:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera["K"], camera["dist"], (w,h), 1, (w,h))
            dst = cv2.undistort(img, camera["K"], camera["dist"], None, newcameramtx)
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]

        # Find markers on a image
        markers, preprocessed_img = markers_to_points(img)

        cv2.imwrite(f'output/test2/camera_{i+1}/preprocessed.jpg', preprocessed_img)

        # Sort the markers
        sorted_markers = find_corresponding_points(markers)
        camera["markers"] = sorted_markers

        # Find the Homeography
        img_points = np.array([ (x, y) for _, x, y in sorted_markers],dtype='double')
        obj_points = np.array([ patterns[id] for id, _, _ in sorted_markers],dtype='double')
        
        new_cameras.append(camera)
    
    calculate_error_pattern(new_cameras, patterns)
    space = plot_cameras_in_space(new_cameras, patterns)

    space.savefig('output/test2/space.jpg')
    
    for i in range(4):
        for j in range(4):
            if i != j:
                F = computeFundamentalMatrix(new_cameras[i]["P"], new_cameras[j]["P"])
                img1, img2, concat = draw_lines(F, new_cameras[i]["img"], new_cameras[j]["img"], new_cameras[i]["markers"], new_cameras[j]["markers"])
                
                os.makedirs(f'output/test2/epipolar/camera_{i+1}_camera_{j+1}', exist_ok = True)
                cv2.imwrite(f'output/test2/epipolar/camera_{i+1}_camera_{j+1}/camera_{i+1}.jpg', img1)
                cv2.imwrite(f'output/test2/epipolar/camera_{i+1}_camera_{j+1}/camera_{j+1}.jpg', img2)
                cv2.imwrite(f'output/test2/epipolar/camera_{i+1}_camera_{j+1}/both.jpg', concat)
    

    ###### kij #######
    print('###### test kij #######')

    new_cameras = []
    patterns = {0: [0, 0, 0], 1: [0, 0, 0.5]}
    cam1 = [(0, 752, 435), (1, 953, 387)]
    cam2 = [(0, 878, 467), (1, 806, 320)]
    cam3 = [(0, 613, 190), (1, 390, 200)]
    cam4 = [(0, 854, 584), (1, 970, 675)]

    cam = [cam1, cam2, cam3, cam4]
    for i, camera in enumerate(cameras):
        
        os.makedirs(f'output/test_kij/camera_{i+1}', exist_ok = True)

        ####### Extrinsic camera calibration #######

        dir_extr = f"data/test_kij/camera_{i+1}.jpg"
        camera["img"] = dir_extr
        img = cv2.imread(dir_extr)

        cv2.imwrite(f'output/test_kij/camera_{i+1}/original.jpg', img)

        # Undistort?
        if undistort:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera["K"], camera["dist"], (w,h), 1, (w,h))
            dst = cv2.undistort(img, camera["K"], camera["dist"], None, newcameramtx)
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]

        # Find markers on a image
        markers, preprocessed_img = markers_to_points(img)

        cv2.imwrite(f'output/test_kij/camera_{i+1}/preprocessed.jpg', preprocessed_img)

        # Sort the markers
        sorted_markers = cam[i]
        camera["markers"] = sorted_markers

        # Find the Homeography
        img_points = np.array([ (x, y) for _, x, y in sorted_markers],dtype='double')
        
        new_cameras.append(camera)
    
    calculate_error_pattern(new_cameras, patterns)

    for i in range(4):
        for j in range(4):
            if i != j:
                F = computeFundamentalMatrix(new_cameras[i]["P"], new_cameras[j]["P"])
                img1, img2, concat = draw_lines(F, new_cameras[i]["img"], new_cameras[j]["img"], new_cameras[i]["markers"], new_cameras[j]["markers"])
                
                os.makedirs(f'output/test_kij/epipolar/camera_{i+1}_camera_{j+1}', exist_ok = True)
                cv2.imwrite(f'output/test_kij/epipolar/camera_{i+1}_camera_{j+1}/camera_{i+1}.jpg', img1)
                cv2.imwrite(f'output/test_kij/epipolar/camera_{i+1}_camera_{j+1}/camera_{j+1}.jpg', img2)
                cv2.imwrite(f'output/test_kij/epipolar/camera_{i+1}_camera_{j+1}/both.jpg', concat)