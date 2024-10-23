import cv2 as cv
import numpy as np

def select_point(frame):
    selected_point = None  # Initialize variable to store the selected point

    # Mouse callback function to get the point
    def get_click(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv.EVENT_LBUTTONDOWN:  # If left mouse button is clicked
            selected_point = (x, y)  # Store the clicked point
            cv.destroyWindow("Select Point")  # Close the window

    # Display the frame and set the mouse callback
    cv.imshow("Select Point", frame)
    cv.setMouseCallback("Select Point", get_click)

    # Wait until the user selects a point
    while selected_point is None:
        if cv.waitKey(1) & 0xFF == ord('q'):  # Allow quitting with 'q'
            cv.destroyAllWindows()
            return None
        cv.waitKey(1)  # Wait for a short moment

    # convert for calcOpticalFlowPyrLK
    selected_point = np.array(selected_point).reshape(-1, 1, 2).astype(np.float32)

    return selected_point  # Return the selected point

def track_point(frame, point_to_track, old_gray):

    # lucas kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    print(f"Old gray shape: {old_gray.shape} and gray frame shape: {gray_frame.shape}")
    print(f"Point to track: {point_to_track}")

    # Calculate optical flow to track the point
    new_point, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, point_to_track, None, **lk_params)

    if st[0][0] == 0:
        print("Point not found")
        return point_to_track, gray_frame

    return new_point, gray_frame

def draw_point(frame, point, color):

    if color == "red":
        rgb = (0, 0, 255)
    elif color == "green":
        rgb = (0, 255, 0)

    # convert to tuple of x, y
    point = tuple(point[0][0].astype(int))

    # Draw a circle around the point
    cv.circle(frame, point, 5, rgb, -1)

    return frame