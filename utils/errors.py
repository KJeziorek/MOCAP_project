from utils.extrinsic import n_view_traingulation
import numpy as np

def calculate_error(img_points, obj_points, P):
    errors = []
    projections = []
    for obj,  img in zip(obj_points,img_points):
        proj = P @ np.append(obj, [1])
        proj /= proj[-1]
        error = (img - proj[:-1])
        projections.append(proj[:-1])
        errors.append(error)

    SSE = np.concatenate(np.power(errors,2)).sum()
    
    return img_points, projections, errors, SSE

def calculate_error_pattern(cameras, patterns):
    errors = []
    max_error = 0
    min_error = np.Inf

    marcer_vec = []
    rec_vec = []

    for i, marker in enumerate(patterns.values()):
        points = []
        P_vec = []

        for cam in cameras:
            P_vec.append(cam["P"])  
            point = (cam["markers"][i][1], cam["markers"][i][2])      
            points.append(point)

        rec = n_view_traingulation(P_vec, points)
        rec = rec[:-1]

        marcer_vec.append(marker)
        rec_vec.append(rec)

        error = np.sqrt(np.power(marker-rec,2).sum())
        max_error = max(error,max_error)
        min_error = min(error, min_error)

        errors.append(error)

    print(f"Traingulation error (for the coplanar marker) are [m]:\n{np.matrix(errors).T}\n\nMean error is: {np.mean(errors)}\n\nBiggest error was: {max_error}\n\nSmallest error was: {min_error}")

    iter = 0
    
    sum_dif = 0
    sum_error = 0

    max_error = 0
    min_error = 999999
    max_diff = 0
    min_diff = 999999

    for i in range(len(marcer_vec)):
        for j in range(i, len(marcer_vec)):
            if i!=j:
                dist_rec = np.sqrt((rec_vec[i][0]-rec_vec[j][0])**2 + (rec_vec[i][1]-rec_vec[j][1])**2 + (rec_vec[i][2]-rec_vec[j][2])**2)
                dist_marker = np.sqrt((marcer_vec[i][0]-marcer_vec[j][0])**2 + (marcer_vec[i][1]-marcer_vec[j][1])**2 + (marcer_vec[i][2]-marcer_vec[j][2])**2)
                
                dif = abs(dist_rec - dist_marker)
                error = np.sqrt(dif**2)

                sum_dif += dif
                sum_error += error

                max_error = max(error, max_error)
                min_error = min(error, min_error)

                max_diff = max(dif, max_diff)
                min_diff = min(dif, min_diff)

                iter += 1
                print(f"Distance betwen point {i} and {j} for marker is {dist_marker} and reconstructed is {dist_rec} dif is {dif} square root error is {error} \n")
    
    print(f"Max dif = {max_diff}, min dif = {min_diff}, mean dif = {sum_dif/iter}, max error = {max_error}, min error = {min_error}, mean error = {sum_error/iter}")

   