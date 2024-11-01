import cv2 as cv
import numpy as np


def triangulate_points(pts1, pts2, R, T):
    """find 3D points from two lists of 2D image points of stereo cameras. The
    coordinate system of the first camera is used as the origin

    Arguments:
    pts1 -- list of image points seen from 1st camera
    pts2 -- list of image points seen from 2nd camera
    R -- rotation matrix to transform from 2nd camera CS to 1st camera CS (see stereo_calibrate)
    T -- translation to transform from 2nd camera CS to 1st camera CS (see stereo_calibrate)
    """
    pts1 = np.array(pts1, dtype=np.float32).T 
    pts2 = np.array(pts2, dtype=np.float32).T
    #print(pts1) -> 2xN

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))]) #origin camera
    #I think this is incorrect since R and T are from 1 to 2 already
    #R_inv = np.linalg.inv(R)
    #T_inv = -np.matmul(R_inv, T)
    #P2 = np.hstack([R_inv, -T_inv])
    P2 = np.hstack([R, T])

    points = cv.triangulatePoints(P1, P2, pts1, pts2).T
    points = np.array([p / p[3] for p in points])

    return points
