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


def stereo_calibrate(imgs1, imgs2, board_shape, square_sz_mm):
    """Perform stereo calibration between two cameras

    Arguments:
    imgs1 -- calibration images of first camera
    imgs2 -- calibration images of second camera. Images should be matched
             with another image taken from the same scene in imgs1 for each index
    board_shape -- tuple (internal corners per column, internal corners per row)
    square_sz_mm -- side length of a chessboard square in mm

    Returns:
    ret -- true if calibration succeded, false otherwise
    R -- rotation matrix from camera 1's coordinate system to camera 2's
         coordinate system
    T -- translation matrix from camera 1's coordinate system to camera 2's
         coordinate system. For a point P given in camera 1's CS, it can be
         represented in camera 2's CS as R*P + T
    E -- essential matrix
    F -- fundamental matrix
    K1 -- camera 1 intrinsic matrix
    E1 -- camera 1 extrinsic matrix
    dist1 -- camera 1 distortion parameters
    K2 -- camera 2 intrinsic matrix
    E2 -- camera 2 extrinsic matrix
    dist2 -- camera 2 distortion parameters
    """
    # get coordinates of board points measured in board's coordinate system
    # that is, [(0, 0, 0), (0, 1, 0), ..., (h, w, 0)]
    grid_pts = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
    grid_pts[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2)
    #? multiply by square size here???
    #?grid_pts *= square_sz_mm
    
    img_size = imgs1[0].shape

    K1, E1, dist1, _ = calibrate(imgs1, board_shape, square_sz_mm)
    K2, E2, dist2, _ = calibrate(imgs2, board_shape, square_sz_mm)

    points1 = []
    points2 = []
    board_pts = []

    for img1, img2 in zip(imgs1, imgs2):
        ret1, corners1 = cv.findChessboardCorners(img1, board_shape, None)
        ret2, corners2 = cv.findChessboardCorners(img2, board_shape, None)

        # chessboard must be found in BOTH images
        if not ret1 or not ret2:
            continue

        # improve precision of corner location
        corners1 = cv.cornerSubPix(img1, corners1, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                                    30, 0.001))
        corners2 = cv.cornerSubPix(img2, corners2, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                                   30, 0.001))

        points1.append(corners1)
        points2.append(corners2)
        board_pts.append(grid_pts)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(board_pts, points1, points2,
                                                     K1, dist1, K2, dist2, img_size,
                                                     criteria=criteria,
                                                     flags=cv.CALIB_FIX_INTRINSIC)

    return ret, R, T, E, F, K1, E1, dist1, K2, E2, dist2
