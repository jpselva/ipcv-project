import cv2 as cv
import numpy as np
from calibration import calibrate, get_calib_images

class VideoProcessor:
    def __init__(self, input_video: str, output_video: str):
        self.input_video = input_video
        self.output_video = output_video

        self.cap = cv.VideoCapture(input_video)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to open video {input_video}")

        self.fps = int(round(self.cap.get(cv.CAP_PROP_FPS)))
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        self.out = cv.VideoWriter(output_video, fourcc, self.fps, (self.frame_width, self.frame_height))

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.cap.release()
        self.out.release()

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

    return selected_point  # Return the selected point

def main(input_videos: list, output_videos: list) -> None:

    if len(input_videos) != len(output_videos): # Check if the number of input and output video files match
        print("Error: The number of input and output video files must match.")
        return

    video_processors = [VideoProcessor(input_video, output_video) for input_video, output_video in zip(input_videos, output_videos)]

    # Calculate the delay based on the highest FPS for all videos
    max_fps = max(vp.fps for vp in video_processors)
    delay = int(1000 / max_fps)
    
    chose_point = False

    # While loop for processing
    while True:

        videos_done = True  

        # Read frames from all video processors and show them
        for video_index, vp in enumerate(video_processors):
            ret, frame = vp.read_frame()

            if ret:

                videos_done = False     # at least one video is not done

                # if it's the first video and there is no selected point, select a point
                if video_index == 0 and chose_point == False:
                    selected_point = select_point(frame)
                    chose_point = True  
                
                # draw point on first video
                if video_index==0 and selected_point is not None:
                    cv.circle(frame, selected_point, 5, (0, 0, 255), -1)  # red circle

                # TODO: Extra processing
            
                cv.imshow(f"Video {video_index}", frame)
                vp.write_frame(frame)
                

        if videos_done:
            break

        # Press Q on keyboard to exit
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release all video processors
    for vp in video_processors:
        vp.release()

    cv.destroyAllWindows()

if __name__ == "__main__":

    # calibrate left camera
    calib_images = get_calib_images("right")
    intrinsic, extrinsics, dist, errors = calibrate(calib_images, (6, 9), 10)
    print(f"K = {intrinsic}")
    print(f"dist = {dist[0]}")
    print(f"extrinsic matrix for 1st image = {extrinsics[0]}")

    input_videos = ["project data/subject1/proefpersoon 1.2_M.avi", "project data/subject1/proefpersoon 1.2_R.avi"]
    output_videos = ["output_videos/output_M.mp4", "output_videos/output_R.mp4"]

    # process videos
    main(input_videos, output_videos)