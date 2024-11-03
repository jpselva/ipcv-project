import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def draw_point(frame, point, color: str):

    # Check if the point is untracked
    if np.array_equal(point, np.array([-1, -1], np.float32)):
        return frame
    
    if color == "red":
        rgb = (0, 0, 255)
    elif color == "green":
        rgb = (0, 255, 0)
    else:    # default color is blue
        rgb = (255, 0, 0)

    # Draw a circle around the point
    cv.circle(frame, point, 5, rgb, -1)

    return frame


def draw_points(frame, points):

    # draw all points green except fifth point, which is red
    for i, point in enumerate(points):
        if i == 4:
            frame = draw_point(frame, point, "red")
        else:
            frame = draw_point(frame, point, "green")

    return frame


def plot_3d_points(points, labels, ax):
    # Unpack the points into x, y, and z coordinates
    x_coords, y_coords, z_coords = zip(*points)

    # Scatter plot
    ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o')

    # Add text labels for each point with its index
    for label, (x, y, z) in zip(labels, points):
        ax.text(x, y, z, label, color='red')

    # Set equal scaling for all axes
    max_range = max(max(x_coords) - min(x_coords),
                    max(y_coords) - min(y_coords),
                    max(z_coords) - min(z_coords)) / 2

    mid_x = (max(x_coords) + min(x_coords)) / 2
    mid_y = (max(y_coords) + min(y_coords)) / 2
    mid_z = (max(z_coords) + min(z_coords)) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.invert_xaxis()


def plot_coordinate_system(x, y, z, origin, ax, scale):
    # Plot each vector with the specified colors
    ax.quiver(*origin, *x, color='red', length=scale, normalize=True, label='X-axis')
    ax.quiver(*origin, *y, color='green', length=scale, normalize=True, label='Y-axis')
    ax.quiver(*origin, *z, color='blue', length=scale, normalize=True, label='Z-axis')

    # Set the aspect ratio and limits if needed for better visualization
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 for x, y, z
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
