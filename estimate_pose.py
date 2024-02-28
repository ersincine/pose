import os
import cv2 as cv
import numpy as np
import torch
from kornia.geometry.conversions import rotation_matrix_to_quaternion


def estimate_pose(img1_no: int, img2_no: int, dataset: str, method: str, save: bool=True, verbose: bool=False) -> tuple[np.ndarray, np.ndarray]:
    path1 = f"datasets/{dataset}/{img1_no:06d}.jpg"
    path2 = f"datasets/{dataset}/{img2_no:06d}.jpg"

    img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if verbose:
        print("Keypoints:", len(kp1), len(kp2))

    if method == "snn":
        ratio = 0.75

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        matches = good

    elif method == "nn":
        bf = cv.BFMatcher()
        matches = bf.match(des1, des2)

    else:
        raise ValueError(f"Unknown method: {method}")

    if verbose:
        print("Matches:", len(matches))

    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    """
    fundamental_matrix, inliers = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)

    points1 = points1[inliers.ravel() == 1]
    points2 = points2[inliers.ravel() == 1]
    """

    """
    # Sitede verilen yaklaşık parametreler
    w = 1920
    h = 1080
    focal_length = 0.7 * w
    principal_point = (w / 2, h / 2)
    K = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ])
    """
    
    # COLMAP'in bulduğu parametreler
    if dataset == "Barn":
        fx = 1152.0
        fy = 1152.0
        cx = 960.0
        cy = 540.0
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    elif dataset == "Truck":
        #raise NotImplementedError  # Maybe the same as Barn?
        # Temporary:
        fx = 1152.0
        fy = 1152.0
        cx = 960.0
        cy = 540.0
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    elif dataset == "Meetingroom":
        #raise NotImplementedError  # Maybe the same as Barn?
        # Temporary:
        fx = 1152.0
        fy = 1152.0
        cx = 960.0
        cy = 540.0
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    # essential_matrix, inliers = cv.findEssentialMat(points1, points2, focal=focal_length, pp=principal_point)
    essential_matrix, inliers = cv.findEssentialMat(points1, points2, cameraMatrix=K, method=cv.USAC_ACCURATE)

    points1 = points1[inliers.ravel() == 1]
    points2 = points2[inliers.ravel() == 1]

    if verbose:
        print("Inliers:", len(points1))

    #assert np.linalg.matrix_rank(essential_matrix) == 2

    #retval, R, t, mask = cv.recoverPose(essential_matrix, points1, points2, focal=focal_length, pp=principal_point)
    retval, R, t, mask = cv.recoverPose(essential_matrix, points1, points2, K)

    # Mask is not 0 or 1, but 0 or 255.

    assert np.isclose(np.linalg.det(R), 1, atol=1e-3)

    num_inliers_after_recover_pose = np.sum(mask != 0)
    assert num_inliers_after_recover_pose == retval

    if verbose:
        print("Inliers after recoverPose:", num_inliers_after_recover_pose)

    t = t.reshape(3)
    q = rotation_matrix_to_quaternion(torch.tensor(R).reshape(1, 3, 3)).reshape(4).numpy()

    if save:
        assert os.path.exists(f"poses/{dataset}/{method}")
        np.savetxt(f"poses/{dataset}/{method}/q-{img1_no}-{img2_no}_estimated.txt", q)
        np.savetxt(f"poses/{dataset}/{method}/t-{img1_no}-{img2_no}_estimated.txt", t)

    return q, t
