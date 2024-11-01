import cv2 as cv
import numpy as np
from calibration import calibrate, get_calib_images, stereo_calibrate
from point_processing import select_point, track_point
from ref import create3dRef
from draw import draw_point, draw3dRef
import matplotlib.pyplot as plt

NUM_REFERENCE_POINTS = 2 #origin and x (for vector x direction) for now

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

        self.frame_count = 0
        self.video_done = False

        self.ref_points = [] #origin and x
        self.ref_point_states = {} #states of reference points

        self.interest_point = None
        self.old_gray = None

        self.out = cv.VideoWriter(output_video, fourcc, self.fps, (self.frame_width, self.frame_height))

    def increment_frame_count(self):
        self.frame_count += 1

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
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    # While loop for processing
    while True:

        # Read frames from all video processors and show them
        for video_index, vp in enumerate(video_processors):
            ret, frame = vp.read_frame()

            if ret:
                # resize frame for better performance (remove for final video)
                frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                
                vp.increment_frame_count()

                #! Point selection
                # first frame
                if vp.frame_count == 1:
                    vp.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # initialize old_gray for tracking

                    #TODO: maybe put this repeated code in a function to avoid repetition
                    origin = select_point(frame, "Origin of referential")
                    x = select_point(frame, "x point for referential")
                    vp.ref_points.append(origin)
                    vp.ref_points.append(x)
                    vp.ref_point_states["origin"] = False #not tracked yet
                    vp.ref_point_states["x"] = False

                # frame for 2s
                if vp.frame_count == 2*vp.fps:
                    vp.interest_point = select_point(frame, "Interest Point")    # select interest point

                #! Point tracking
                if len(vp.ref_points) == NUM_REFERENCE_POINTS: #if all reference points are selected
                    for point in vp.ref_points:
                        point = np.array(point).reshape(-1, 1, 2).astype(np.float32) #correct shape for tracking                        
                        point, gray_frame = track_point(frame, point, vp.old_gray)
                        frame = draw_point(frame, point, "green")

                if vp.interest_point is not None:
                    vp.interest_point, gray_frame = track_point(frame, vp.interest_point, vp.old_gray)
                    frame = draw_point(frame, vp.interest_point, "red")

                #TODO: if all the reference points are being tracked for all videos [tracking states not implemented yet]
                #if all(vp.reference_points_states.values() for vp in video_processors):
                #! IN PROGRESS: Create 3d referential (for now just tracking x and origin) 
                if(all(len(vp.ref_points) == NUM_REFERENCE_POINTS for vp in video_processors)):
                    origin3d, x_vector, y_vector, z_vector = create3dRef(frame, video_processors[0].ref_points,
                                                            video_processors[1].ref_points, R2_1, T2_1)
                    draw3dRef(ax, origin3d, x_vector, y_vector, z_vector)
                    plt.draw()
                    plt.pause(0.01)


                vp.old_gray = gray_frame

                cv.imshow(f"Video {video_index}", frame)
                vp.write_frame(frame)
            
            else:
                vp.video_done = True

        if all(vp.video_done for vp in video_processors):   # if all videos are done
            break

        # Press Q on keyboard to exit
        if cv.waitKey(delay) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            plt.close(fig)
            break

    # Release all video processors
    for vp in video_processors:
        vp.release()

    cv.destroyAllWindows()
    plt.close(fig)

if __name__ == "__main__":

    # Get calibration images for both left and right cameras
    calib_images_left = get_calib_images("left")
    calib_images_right = get_calib_images("right")

    board_shape = (6, 9)
    square_sz_mm = 10 

    ret, R2_1, T2_1, E, F, K1, E1, dist1, K2, E2, dist2 = stereo_calibrate(calib_images_left, calib_images_right, board_shape, square_sz_mm)
    
    if ret:
        print("Stereo calibration successful.")
        print(f"K1 = {K1}")  # Intrinsic matrix for camera 1
        print(f"K2 = {K2}")  # Intrinsic matrix for camera 2
        print(f"R (from 1 to 2)= {R2_1}")    # Rotation matrix from camera 1 to camera 2
        print(f"T (from 1 to 2)= {T2_1}")    # Translation vector from camera 1 to camera 2
    else:
        print("Stereo calibration failed.")

    input_videos = ["project data/subject1/proefpersoon 1.2_M.avi", "project data/subject1/proefpersoon 1.2_R.avi"]
    output_videos = ["output_videos/output_M.mp4", "output_videos/output_R.mp4"]

    # Process videos
    main(input_videos, output_videos)
