import cv2 as cv
import numpy as np
from calibration import get_calib_images, stereo_calibrate
from point_processing import select_point, track_points, select_n_points
from ref import create3dRef, convertToRef
from draw import plot_3d_points, plot_coordinate_system, draw_points
from triangulation import triangulate_points
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.raise_window'] = False  # so matplotlib won't put window in focus whenever we update it

SELECT_POINTS = False


def show_multiple_frames(frames, title, dimensions):
    w, h = dimensions
    combined = np.hstack(frames)
    resized = cv.resize(combined, (w, h))
    cv.imshow(title, resized)


if __name__ == "__main__":

    # get calibration images
    calib_images_middle = get_calib_images("middle")
    calib_images_right = get_calib_images("right")
    calib_images_left = get_calib_images("left")

    # perform stereo calibration
    board_shape = (6, 9)
    square_sz_mm = 10
    ret, R2, T2, _, _, K1, _, _, K2, _, _ = stereo_calibrate(calib_images_middle, calib_images_right, board_shape, square_sz_mm)
    ret, R3, T3, _, _, _, _, _, K3, _, _ = stereo_calibrate(calib_images_middle, calib_images_left, board_shape, square_sz_mm)

    input_videos = ["project data/subject1/proefpersoon 1.2_M.avi",
                    "project data/subject1/proefpersoon 1.2_R.avi",
                    "project data/subject1/proefpersoon 1.2_L.avi"]

    cap_m = cv.VideoCapture(input_videos[0])
    cap_r = cv.VideoCapture(input_videos[1])
    cap_l = cv.VideoCapture(input_videos[2])

    ret_m, frame_m = cap_m.read()
    ret_r, frame_r = cap_r.read()
    ret_l, frame_l = cap_l.read()

    fps = int(round(cap_m.get(cv.CAP_PROP_FPS)))
    frame_width = int(cap_m.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_m.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count_m = 0

    # these are hardcoded points I found that matched the desired features.
    # use select_n_points below to allow the user to select the points
    # they must be selected in the right order: nose, cheek_l, forehead, cheek_r
    points_m = np.array([[528, 430],
                         [646, 399],
                         [512, 314],
                         [383, 407]], np.float32)

    points_r = np.array([[742, 443],
                         [895, 409],
                         [743, 320],
                         [618, 419]], np.float32)

    points_l = np.array([[324, 460],
                         [421, 429],
                         [292, 337],
                         [148, 433]], np.float32)

    if SELECT_POINTS:
        points_m = np.array(select_n_points(frame_m, ["nose", "right cheek", "forehead", "left cheek"]), np.float32)
        points_r = np.array(select_n_points(frame_r, ["nose", "right cheek", "forehead", "left cheek"]), np.float32)
        points_l = np.array(select_n_points(frame_l, ["nose", "right cheek", "forehead", "left cheek"]), np.float32)

    points = triangulate_points(points_m, points_r, R2, T2, K1, K2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    while True:
        ret_m, next_frame_m = cap_m.read()
        ret_r, next_frame_r = cap_r.read()
        ret_l, next_frame_l = cap_l.read()
        frame_count_m += 1

        if not ret_m or not ret_r:
            print("couldn't read frame")
            break

        # if user clicks s, reselect interest point
        if cv.waitKey(2) & 0xFF == ord("s"):

            # if one of the arrays still has interest point, remove it
            if len(points_m) == 5:
                points_m = np.delete(points_m, 4, 0)
            if len(points_r) == 5:
                points_r = np.delete(points_r, 4, 0)
            if len(points_l) == 5:
                points_l = np.delete(points_l, 4, 0)

            interest_point_m = np.array(select_point(frame_m, "interest point")).astype(np.float32)
            interest_point_r = np.array(select_point(frame_r, "interest point")).astype(np.float32)
            interest_point_l = np.array(select_point(frame_l, "interest point")).astype(np.float32)
            # add the interest point
            if interest_point_m is not None and interest_point_r is not None and interest_point_l is not None:
                points_m = np.vstack((points_m, interest_point_m))
                points_r = np.vstack((points_r, interest_point_r))
                points_l = np.vstack((points_l, interest_point_l))

        # pause video if user presses 'p'
        if (cv.waitKey(1) & 0xFF == ord("p")):
            while True:
                if (cv.waitKey(1) & 0xFF == ord("p")):
                    break

        # track each point to the current frame
        points_m = track_points(next_frame_m, frame_m, points_m)
        points_r = track_points(next_frame_r, frame_r, points_r)
        points_l = track_points(next_frame_l, frame_l, points_l)

        # if interest point is lost, remove it
        if len(points_m) == 5 and len(points_r) == 5 and len(points_l) == 5:
            if (np.array_equal(points_m[4], np.array([-1, -1], np.float32)) or
               np.array_equal(points_r[4], np.array([-1, -1], np.float32)) or
               np.array_equal(points_l[4], np.array([-1, -1], np.float32))):
                points_m = np.delete(points_m, 4, 0)
                points_r = np.delete(points_r, 4, 0)
                points_l = np.delete(points_l, 4, 0)
                print("Interest point lost, press 's' to reselect")

        # points triangulation
        points = triangulate_points(points_m, points_r, R2, T2, K1, K2)
        points_alt = triangulate_points(points_m, points_l, R3, T3, K1, K3)

        # error is RMS of distances between triangulated points from each camera
        error = np.sqrt(np.mean(np.power(points - points_alt, 2 * np.ones(points.shape))))

        face_points = {'nose': points[0],
                       'cheek_r': points[1],
                       'forehead': points[2],
                       'cheek_l': points[3]}

        if len(points) == 5:
            face_points['interest'] = points[4]

        # calculate face coordinate system
        origin, x, y, z = create3dRef(face_points)

        # update 3d plot
        ax.cla()
        plot_3d_points(face_points.values(), face_points.keys(), ax)
        plot_coordinate_system(x, y, z, origin, ax, 50)
        plt.draw()
        plt.pause(0.0001)

        # draw dots and show cameras
        frame_m = draw_points(frame_m, np.int32(points_m))
        frame_r = draw_points(frame_r, np.int32(points_r))
        frame_l = draw_points(frame_l, np.int32(points_l))
        show_multiple_frames([frame_l, frame_m, frame_r], "Videos L, M, R",
                             [3 * frame_width // 3, frame_height // 3])

        # print interest point in face coordinate system and current error
        if face_points.get('interest') is not None:
            interest_point = convertToRef(face_points['interest'], origin, x, y, z)
            ix, iy, iz = interest_point
            print(f"Interest point: ({ix} {iy} {iz}) mm, error = {error} mm")
        else:
            print(f"error = {error} mm")

        cv.waitKey(1)
        frame_m = next_frame_m
        frame_r = next_frame_r
        frame_l = next_frame_l
