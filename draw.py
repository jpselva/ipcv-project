import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def draw_point(frame, point: tuple[int, int], color: str):

    if color == "red":
        rgb = (0, 0, 255)
    elif color == "green":
        rgb = (0, 255, 0)
    else:    # default color is blue
        rgb = (255, 0, 0)

    # Draw a circle around the point
    cv.circle(frame, point, 5, rgb, -1)

    return frame

def drawVector(frame, origin, dest, color):
    if color == "red":
        rgb = (0, 0, 255)
    elif color == "green":
        rgb = (0, 255, 0)
    else:    # default color is blue
        rgb = (255, 0, 0)
    
    # convert to tuple of x, y
    origin = tuple(origin[0][0].astype(int))
    dest = tuple(dest[0][0].astype(int))
    
    # Draw an arrow from origin to dest
    cv.arrowedLine(frame, origin, dest, rgb, 2)
    
    return frame


def draw3dRef(ax: plt.Axes, origin, x, y, z):
  ax.cla() # clear the current axes
  # DEBUG
  print("Origin:", origin)
  print("X:", x)
  print("Y:", y)
  print("Z:", z)

  arrow_size = 0.1  
  x_end = origin + arrow_size * x / np.linalg.norm(x)
  y_end = origin + arrow_size * y / np.linalg.norm(y)
  z_end = origin + arrow_size * z / np.linalg.norm(z)
  
  #draw camera origin
  ax.scatter(0, 0, 0, color='black', label='Main Camera Pinhole')
  ax.text(0, 0, 0, "Camera Pinhole", color='black')
  #draw unit vectors
  ax.quiver(origin[0], origin[1], origin[2], x_end[0], x_end[1], x_end[2], arrow_length_ratio=0.01, color='b', label='x')
  ax.quiver(origin[0], origin[1], origin[2], y_end[0], y_end[1], y_end[2], arrow_length_ratio=0.01, color='g', label='y')
  ax.quiver(origin[0], origin[1], origin[2], z_end[0], z_end[1], z_end[2], arrow_length_ratio=0.01, color ='r', label='z')
  ax.text(x_end[0], x_end[1], x_end[2], "face x", color='b')
  ax.text(y_end[0], y_end[1], y_end[2], "face y", color='g')
  ax.text(z_end[0], z_end[1], z_end[2], "face z", color='r')

  ax.set_box_aspect([1,1,1])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Face 3D Referential viewed from main camera')