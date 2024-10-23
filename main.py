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

def main(input_videos: list, output_videos: list) -> None:

    if len(input_videos) != len(output_videos): # Check if the number of input and output video files match
        print("Error: The number of input and output video files must match.")
        return

    video_processors = [VideoProcessor(input_video, output_video) for input_video, output_video in zip(input_videos, output_videos)]

    # Calculate the delay based on the highest FPS for all videos
    max_fps = max(vp.fps for vp in video_processors)
    delay = int(1000 / max_fps)

    # While loop for processing
    while True:

        videos_done = True  

        # Read frames from all video processors and show them
        for i, vp in enumerate(video_processors):
            ret, frame = vp.read_frame()

            if ret:
                # TODO: Process the frame here if needed (e.g., resizing, filtering, etc.)
            
                cv.imshow(f"Video {i}", frame)
                vp.write_frame(frame)
                videos_done = False     # at least one video is not done

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