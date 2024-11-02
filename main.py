import cv2 as cv
import numpy as np
from calibration import get_calib_images, stereo_calibrate
from point_processing import select_point, track_points
from ref import create3dRef, convertToRef
from draw import plot_3d_points, plot_coordinate_system, draw_points
from triangulation import triangulate_points
import matplotlib.pyplot as plt


def select_n_points(frame, labels):
    points = []

    for label in labels:
        points.append(select_point(frame, label))

    return points


def show_multiple_frames(frames, title, dimensions):
    w, h = dimensions
    combined = np.hstack(frames)
    resized = cv.resize(combined, (w, h))
    cv.imshow(title, resized)


if __name__ == "__main__":

    # Get calibration images for both left and right cameras
    calib_images_left = get_calib_images("left")
    calib_images_right = get_calib_images("right")

    board_shape = (6, 9)
    square_sz_mm = 10
    ret, R, T, E, F, K1, E1, dist1, K2, E2, dist2 = stereo_calibrate(calib_images_left, calib_images_right, board_shape, square_sz_mm)

    input_videos = ["project data/subject1/proefpersoon 1.2_M.avi", "project data/subject1/proefpersoon 1.2_R.avi"]
    output_videos = ["output_videos/output_M.mp4", "output_videos/output_R.mp4"]

    cap_m = cv.VideoCapture(input_videos[0])
    cap_r = cv.VideoCapture(input_videos[1])

    fps = int(round(cap_m.get(cv.CAP_PROP_FPS)))

    ret_m, frame_m = cap_m.read()
    fps_m = int(round(cap_m.get(cv.CAP_PROP_FPS)))
    frame_width = int(cap_m.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_m.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count_m = 0

    ret_r, frame_r = cap_r.read()

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

    # points_m = np.array(select_n_points(frame_m, ["nose", "right cheek", "forehead", "left cheek"]), np.float32)
    # points_r = np.array(select_n_points(frame_r, ["nose", "right cheek", "forehead", "left cheek"]), np.float32)

    points = triangulate_points(points_m, points_r, R, T, K1, dist1, K2, dist2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    while True:
        ret_m, next_frame_m = cap_m.read()
        ret_r, next_frame_r = cap_r.read()
        frame_count_m += 1

        if not ret_m or not ret_r:
            print("couldn't read frame")
            break

        if (frame_count_m == 2 * fps_m):  # tongue appears at around 2 seconds
            interest_point_m = np.array(select_point(frame_m, "interest point")).astype(np.float32)
            interest_point_r = np.array(select_point(frame_r, "interest point")).astype(np.float32)
            # add the interest point
            points_m = np.vstack((points_m, interest_point_m))
            points_r = np.vstack((points_r, interest_point_r))

        points_m = track_points(next_frame_m, frame_m, points_m)
        points_r = track_points(next_frame_r, frame_r, points_r)

        points = triangulate_points(points_m, points_r, R, T, K1, dist1, K2, dist2)

        face_points = {'nose': points[0],
                       'cheek_r': points[1],
                       'forehead': points[2],
                       'cheek_l': points[3]}

        if len(points) == 5:
            face_points['interest'] = points[4]

        origin, x, y, z = create3dRef(face_points)

        ax.cla()
        plot_3d_points(face_points.values(), face_points.keys(), ax)
        plot_coordinate_system(x, y, z, origin, ax, 50)

        # DRAW DOTS and SHOW CAMERAS
        frame_m = draw_points(frame_m, np.int32(points_m), "green")
        frame_r = draw_points(frame_r, np.int32(points_r), "green")
        show_multiple_frames([frame_m, frame_r], "Videos M, R",
                             [2 * frame_width // 3, frame_height // 3])

        # UPDATE 3D PLOT
        plt.draw()
        plt.pause(1.0 / fps)

        # PAUSE VIDEO
        if (cv.waitKey(1) & 0xFF == ord("p")):
            while True:
                if (cv.waitKey(1) & 0xFF == ord("p")):
                    break

        cv.waitKey(1)
        frame_m = next_frame_m
        frame_r = next_frame_r

        # PRINTS
        if face_points.get('interest') is not None:
            interest_point = convertToRef(face_points['interest'], origin, x, y, z)
            print("Interest point: [", interest_point[0], interest_point[1], interest_point[2], "] mm")
