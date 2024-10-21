import cv2 as cv
import numpy as np
import os


def get_calib_images(camera):

    if camera == "left":
        calib_dir = "project data/Calibratie 1/calibrationLeft"
    elif camera == "right":
        calib_dir = "project data/Calibratie 1/calibrationRight"
    elif camera == "middle":
        calib_dir = "project data/Calibratie 1/calibrationMiddle"
    calib_img_paths = []
    for f in os.listdir(calib_dir):
        full_path = os.path.join(calib_dir, f)
        if (os.path.isfile(full_path)):
            calib_img_paths.append(full_path)

    calib_images = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in calib_img_paths]

    return calib_images

def calibrate(imgs, board_shape, square_sz_mm):
    """Return camera calibration parameters

    Arguments:
    imgs -- chessboard calibration images
    board_shape -- tuple (internal corners per column, internal corners per row)
    square_sz_mm -- side length of a chessboard square in mm

    Returns:
    intrinsic -- 3x3 intrinsic camera matrix
    extrinsics -- 4x4 extrinsic camera matrices for each image in which the
                  chessboard was detected
    dist -- distortion parameters (see docs for cv.calibrateCamera)
    errors -- reprojection errors for each image in which the chessboard was
              detected
    """
    # get coordinates of board points measured in board's coordinate system
    # that is, [(0, 0, 0), (0, 1, 0), ..., (h, w, 0)]
    grid_pts = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
    grid_pts[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2)

    boardpoints = []
    imgpoints = []
    for img in imgs:
        ret, corners = cv.findChessboardCorners(img, board_shape, None)
        if not ret:
            continue
        # improve precision of corner location
        corners = cv.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                  (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                                   30, 0.001))
        imgpoints.append(corners)
        boardpoints.append(grid_pts)

    ret, intrinsic, dist, rvecs, tvecs, _, _, errors = cv.calibrateCameraExtended(boardpoints, imgpoints, img.shape[::-1], None, None)

    extrinsics = np.array([transform_from_rot_trans(r, t) for r, t in zip(rvecs, tvecs)])
    extrinsics[:, :3, :3] /= square_sz_mm

    return intrinsic, extrinsics, dist, errors

def transform_from_rot_trans(rvec, tvec):
    """Return 4x4 transformation matrix from rotation vector and translation vector """
    rmat = cv.Rodrigues(rvec, None)[0]
    transf = np.concatenate([rmat, tvec], axis=1)
    transf = np.concatenate([transf, np.array([[0, 0, 0, 1]])])
    return transf
