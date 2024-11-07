import cv2 as cv
import numpy as np

def select_n_points(frame, labels):
    points = []

    for label in labels:
        points.append(select_point(frame, label))

    return points

def select_point(frame, point_name="Point"):
    selected_point = None  # variable to store the selected point
    
    # mouse callback function to get the point
    def get_click(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv.EVENT_LBUTTONDOWN:  # if left mouse button is clicked
            selected_point = (x, y)
            print("Selected point:", selected_point)
            
    #display the frame and set the mouse callback
    cv.imshow(f"Select {point_name}", frame)
    try:
        cv.setMouseCallback(f"Select {point_name}", get_click)
    except Exception as e:
        print(f"Error setting mouse callback: {e}")
        
    # wait until the user selects a point
    while selected_point is None:
        if cv.waitKey(1) & 0xFF == ord('q'):  #allow quitting with 'q'
            cv.destroyAllWindows()
            return None
        cv.waitKey(10)
        
    cv.destroyWindow(f"Select {point_name}")
    return selected_point  

#not being used
""" def track_point(frame, point_to_track, old_gray):

    # lucas kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow to track the point
    new_point, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, point_to_track, None, **lk_params)

    new_point = tuple(map(int, new_point.flatten())) #convert to tuple of x,y coordinates

    if st[0][0] == 0:
        print("Point not found")
        return None, gray_frame

    return new_point, gray_frame """


def track_points(frame, prev_frame, points: np.ndarray):
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    new_points, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)

    # define threshold for movement
    movement_threshold = 15  # Example threshold, adjust as needed

    for i in range(len(st)):
        if st[i][0] == 0 or np.linalg.norm(new_points[i] - points[i]) > movement_threshold:

            print(f"Point {i} not found or moved too much")
            new_points[i] = np.array([-1, -1], np.float32)  # Mark as lost

    return new_points
