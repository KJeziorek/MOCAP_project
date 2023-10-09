import numpy as np
import cv2
from sklearn.metrics import r2_score
from typing import List, Tuple


def markers_to_points(frame: np.array) -> List[Tuple[int, float, float]]:
 """ Find markers and return (id,x,y) after segmentation """

 # 1. Convert to gray scale
 try:
  g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 except:
  g_frame = frame

 # TODO - for real image add more preprocessing
 blur = cv2.GaussianBlur(g_frame,(5,5),0)
 # 2. Perform thresholding
 _, thr = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

 # 3. Perform labeling
 connectivity = 8
 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr, connectivity, cv2.CV_32S)

 # 4. Create object list
 obj = []
 for id, (x, y) in enumerate(centroids[1:]):
  obj.append((id, x, y))

 return obj, thr

def check_first_pair_side(intersection, separate, z_first, z_second) -> bool:
    """Return True if first pair is left side"""
    """Return False if first pair is right side"""

    x1, y1 = intersection
    x2, y2 = separate
    if z_first[0] > 0 and z_second[0] > 0:
        if z_first[0]*x2 + z_first[1] > y2:
            return True
        else:
            return False
    elif z_first[0] < 0 and z_second[0] < 0:
        if z_first[0]*x2 + z_first[1] > y2:
            return False
        else:
            return True
    else:
        if z_first[0]*x2 + z_first[1] > y2 and z_second[0]*x2 + z_second[1] > y2:
            if z_first[0] > 0:
                return True
            else:
                return False
        elif z_first[0]*x2 + z_first[1] < y2 and z_second[0]*x2 + z_second[1] < y2:

            if z_first[0] > 0:
                return True
            else:
                return False

        elif z_first[0]*x2 + z_first[1] < y2 and  z_first[0] < 0:
            return True
        elif z_first[0]*x2 + z_first[1] < y2 and  z_first[0] > 0:
            return False
        elif z_first[0]*x2 + z_first[1] > y2 and  z_first[0] < 0:
            return True
        elif z_first[0] * x2 + z_first[1] > y2 and z_first[0] > 0:
            return False


def find_corresponding_points(vec) -> List[Tuple[int, float, float]]:
    """Sort points based on marker shape"""
    pairs = []
    r2_vec = []
    z_vec = []
    correct_idx = []

    # Make pairs of 3 elements, fit linear regression and calculate r2 score
    for i in range(6):
        for j in range(i, 6):
            for k in range(j, 6):
                if i != j and i != k and j != k:
                    x = np.array([vec[i][1], vec[j][1], vec[k][1]])
                    y = np.array([vec[i][2], vec[j][2], vec[k][2]])

                    z = np.polyfit(x, y, 1)
                    r2_vec.append(r2_score(y, z[0] * x + z[1]))
                    pairs.append([i, j, k])
                    z_vec.append(z)

    # Get first and second pair and parameters of linear regression based on best r2 score metric
    first_pair = pairs.pop(np.argmax(r2_vec))
    z_first = z_vec.pop(np.argmax(r2_vec))

    r2_vec.pop(np.argmax(r2_vec))

    second_pair = pairs.pop(np.argmax(r2_vec))
    z_second = z_vec.pop(np.argmax(r2_vec))

    # intersection is point of intersection of two straight lines
    intersection = [value for value in first_pair if value in second_pair]

    # separate is point that has no straight line
    separate = [value for value in [0, 1, 2, 3, 4, 5] if value not in first_pair + second_pair]

    # check if first pair of points belong to left or right side
    side = check_first_pair_side(vec[intersection[0]][1:], vec[separate[0]][1:], z_first, z_second)

    # append intersection and separate point to results
    correct_idx.append((0, vec[intersection[0]][1], vec[intersection[0]][2]))
    correct_idx.append((3, vec[separate[0]][1], vec[separate[0]][2]))

    if side:
        # side == 1 -> first pair belong to left side, second pair belong to right side
        left_pair = first_pair
        right_pair = second_pair
    else:
        # side == 0 -> second pair belong to left side, first pair belong to right side
        left_pair = second_pair
        right_pair = first_pair

    # For right side pair calculate which point is closer to intersection point
    right_pair.remove(intersection[0])

    idx1 = right_pair.pop()
    distance_1 = np.sqrt((vec[intersection[0]][1] - vec[idx1][1]) ** 2 + (vec[intersection[0]][2] - vec[idx1][2]) ** 2)

    idx2 = right_pair.pop()
    distance_2 = np.sqrt((vec[intersection[0]][1] - vec[idx2][1]) ** 2 + (vec[intersection[0]][2] - vec[idx2][2]) ** 2)

    if distance_1 > distance_2:
        correct_idx.append((1, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((2, vec[idx1][1], vec[idx1][2]))
    else:
        correct_idx.append((2, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((1, vec[idx1][1], vec[idx1][2]))

    # For left side pair calculate which point is closer to intersection point

    left_pair.remove(intersection[0])

    idx1 = left_pair.pop()
    distance_1 = np.sqrt((vec[intersection[0]][1] - vec[idx1][1]) ** 2 + (vec[intersection[0]][2] - vec[idx1][2]) ** 2)

    idx2 = left_pair.pop()
    distance_2 = np.sqrt((vec[intersection[0]][1] - vec[idx2][1]) ** 2 + (vec[intersection[0]][2] - vec[idx2][2]) ** 2)

    if distance_1 > distance_2:
        correct_idx.append((5, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((4, vec[idx1][1], vec[idx1][2]))
    else:
        correct_idx.append((4, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((5, vec[idx1][1], vec[idx1][2]))

    # Sort values based on index
    return sorted(correct_idx, key=lambda tup: tup[0])