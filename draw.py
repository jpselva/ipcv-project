import cv2 as cv

def draw_point(frame, point, color):

    if color == "red":
        rgb = (0, 0, 255)
    elif color == "green":
        rgb = (0, 255, 0)
    else:    # default color is blue
        rgb = (255, 0, 0)

    # convert to tuple of x, y
    point = tuple(point[0][0].astype(int))

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
