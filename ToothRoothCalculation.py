import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math


def calculate_pca(points):
    points = np.array(points)
    pca = PCA(n_components=2)
    points_pca = pca.fit(points).transform(points)
    return points_pca


def pca_calculation(r):
    pca_min_max = []
    for i, points in enumerate(r):
        p = np.array(points)
        pca = PCA(n_components=1)
        pca.fit(p)
        y_pca = pca.transform(p)
        y_new = pca.inverse_transform(y_pca)
        projected_points = y_new.tolist()
        projected_points.sort(key=lambda x: x[1])

        max_point = projected_points[0]
        min_point = projected_points[-1]

        pca_min_max.append([np.array(min_point, dtype=np.int0), np.array(max_point, dtype=np.int0)])
    return pca_min_max


def pca_calculation_visualize(contour, rotated_bbox):
    plt.figure()
    for i, points in enumerate(contour):
        p = np.array(points)
        pca = PCA(n_components=1)
        pca.fit(p)
        y_pca = pca.transform(p)
        y_new = pca.inverse_transform(y_pca)

        ax = plt.subplot(4, 7, i + 1)
        plt.scatter(p[:, 0], p[:, 1], s=2, zorder=10, alpha=0.3)
        plt.scatter(y_new[:, 0], y_new[:, 1], s=2, zorder=10, alpha=0.8)
        # plt.title(f"Tooth {i}")
        # for length, vector in zip(pca.explained_variance_, pca.components_):
        #     v = vector * 3 * np.sqrt(length)
        #     draw_vector(pca.mean_, pca.mean_ + v)
        xs = [p[0] for p in rotated_bbox[i]]
        xs.append(rotated_bbox[i][0][0])
        ys = [p[1] for p in rotated_bbox[i]]
        ys.append(rotated_bbox[i][0][1])
        plt.plot(xs, ys, color='red', linestyle='dashed')

        plt.axis('equal')
        plt.gca().invert_yaxis()
    plt.show()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def crop_rect(img, b):
    # get the parameter of the small rectangle
    # print("box:", b)
    b = np.array(b)
    rect = cv2.minAreaRect(b)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)

    center, size, angle = rect[0], rect[1], rect[2]
    # print("rotations angle: ", angle)
    # ### vorsicht!
    # if not -45 < angle < 45:
    #     angle = 90 -angle
    #     print("new angle: ", angle)
    #     size = reversed(size)

    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def extract_tooth_images(b, image):
    # print("box:", b)
    b = np.array(b)
    rect = cv2.minAreaRect(b)
    # print("rect: ", rect)
    width = int(rect[1][0])
    height = int(rect[1][1])

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def clockwiseangle_and_distance(point, origin, refvec):
    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def sort_tooth_lrtb(r, separate=False):
    arr = np.array(r)
    sorted_arr = sorted(arr, key=lambda x: np.mean(x, axis=0)[0], reverse=False)
    ok_array = []
    uk_array = []
    mid_point = np.mean(np.array([np.mean(x, axis=0) for x in r]), axis=0)
    for c in sorted_arr:
        if (np.mean(c, axis=0)[1] - mid_point[1]) < 0:
            ok_array.append(c)
        else:
            uk_array.append(c)
    sorted_array = ok_array + uk_array
    if separate:
        return ok_array, uk_array
    else:
        return sorted_array


def sqrt_distance(p, q):
    return math.sqrt(sum((s1 - s2) ** 2 for s1, s2 in zip(p, q)))


def calculate_relative_point(contour, rois, rbox):
    sorted_relative_to_top_left_upper = []
    sorted_relative_to_bottom_left_lower = []

    # contour = result['contours']
    sorted_tooth_boundary_upper, sorted_tooth_boundary_lower = sort_tooth_lrtb(contour, separate=True)
    # rois = result['rois']
    bboxes = [[[roi[1], roi[0]], [roi[3], roi[2]]] for roi in rois]
    sorted_roi_upper, sorted_roi_lower = sort_tooth_lrtb(bboxes, separate=True)
    # rbox = result['rbox']
    sorted_rotated_box_upper, sorted_rotated_box_lower = sort_tooth_lrtb(rbox, separate=True)

    sorted_pca_min_max_upper = pca_calculation(sorted_tooth_boundary_upper)
    sorted_pca_min_max_lower = pca_calculation(sorted_tooth_boundary_lower)

    for i in range(len(sorted_pca_min_max_upper)):

        # finding nearst point of rotated box to roi's left-bottom point
        # there is no math.dist() function in python version below 3.8! so ...
        # finding top_right point of rotated box
        sorted_rbox_top_left = sort_points_clockwise(sorted_rotated_box_upper[i])[0]
        # finding relative position to top left point
        pca_max = sorted_pca_min_max_upper[i][1]
        relative_pos_tl = [int(sqrt_distance(sorted_rbox_top_left, pca_max)),
                        int(sqrt_distance(sorted_pca_min_max_upper[i][0], sorted_pca_min_max_upper[i][1]))]
        print("relative to top left: ", relative_pos_tl, "Coordinate: ", sorted_rbox_top_left)
        sorted_relative_to_top_left_upper.append(relative_pos_tl)

    for i in range(len(sorted_pca_min_max_lower)):
        # finding nearst point of rotated box to roi's left-bottom point
        # there is no math.dist() function in python version below 3.8! so ...
        # finding top_right point of rotated box
        sorted_rbox_bottom_left = sort_points_clockwise(sorted_rotated_box_lower[i])[-1]
        # finding relative position to lb
        pca_min = sorted_pca_min_max_upper[i][0]
        relative_pos_bl = [int(sqrt_distance(sorted_rbox_bottom_left, sorted_pca_min_max_lower[i][0])),
                           int(sqrt_distance(sorted_pca_min_max_lower[i][0], sorted_pca_min_max_lower[i][1]))]
        print("relative to bottom left: ", relative_pos_bl, "Coordinate: ", sorted_rbox_bottom_left)
        sorted_relative_to_bottom_left_lower.append(relative_pos_bl)
    sorted_relative_to_lb = sorted_relative_to_top_left_upper + sorted_relative_to_bottom_left_lower
    return sorted_relative_to_lb

def sort_points_clockwise(points):
    centroid = np.average(np.array(points), axis=0)
    upper_points = []
    lower_points = []
    for point in points:
        if(centroid[1] - point[1] >=0):
            upper_points.append(point)
        else:
            lower_points.append(point)

    upper_points.sort(key=lambda x: x[0])
    lower_points.sort(key=lambda y: y[0])
    sorted_bbox = [upper_points[0], upper_points[1], lower_points[1], lower_points[0]]
    return sorted_bbox