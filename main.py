import os
import cv2 as cv
from calibration import calibrate


if __name__ == '__main__':
    calib_dir_r = "project data/Calibratie 1/calibrationRight"
    calib_img_paths = []
    for f in os.listdir(calib_dir_r):
        full_path = os.path.join(calib_dir_r, f)
        if (os.path.isfile(full_path)):
            calib_img_paths.append(full_path)

    calib_images = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in calib_img_paths]
    intrinsic, extrinsics, dist, errors = calibrate(calib_images, (6, 9), 10)
    print(f"K = {intrinsic}")
    print(f"dist = {dist[0]}")
    print(f"extrinsic matrix for 1st image = {extrinsics[0]}")
