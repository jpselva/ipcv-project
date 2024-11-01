import cv2 as cv
import numpy as np


def triangulate_points(pts1, pts2, R, T, K1, dist1, K2, dist2):
    """find 3D points from two lists of 2D image points of stereo cameras. The
    coordinate system of the first camera is used as the origin

    Arguments:
    pts1 -- list of image points seen from 1st camera
    pts2 -- list of image points seen from 2nd camera
    R -- rotation matrix to transform from 1st camera CS to 2nd camera CS (see stereo_calibrate)
    T -- translation to transform from 1st camera CS to 2nd camera CS (see stereo_calibrate)
    K1 -- intrisic matrix of camera 1
    dist1 -- distortion coefficients of camera 1
    K2 -- intrinsic matrix of camera 2
    dist2 -- distortion coefficients of camera 2
    """
    # P1 is the identity projection matrix because camera 1 is origin
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    R_inv = np.linalg.inv(R)
    T_inv = -np.matmul(R_inv, T)
    P2 = np.hstack([R_inv, T_inv])

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    pts1 = cv.undistortPoints(pts1, K1, dist1).reshape(-1, 2)
    pts2 = cv.undistortPoints(pts2, K2, dist2, P=P2).reshape(-1, 2)

    points = cv.triangulatePoints(P1, P2, pts1.T, pts2.T).T

    # convert back from homogenous coordinates
    points = np.array([(p / p[3])[:3] for p in points])

    return points
