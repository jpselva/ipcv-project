import cv2 as cv
from calibration import calibrate, get_calib_images

def main(input_video_file: str, output_video_file: str) -> None:

    cap = cv.VideoCapture(input_video_file)

    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_file}")
        return

    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    delay = int(1000 / fps)  # calculate delay in milliseconds based on fps

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()

        # process frame
        if ret:

            #TODO: process frame here
            
            # write frame that you processed to output
            out.write(frame)
            cv.imshow('Frame', frame)  # display the frame

            # Press Q on keyboard to exit
            if cv.waitKey(delay) & 0xFF == ord('q'):
                break

        else:
            break
        

if __name__ == "__main__":

    # calibrate left camera
    calib_images = get_calib_images("right")
    intrinsic, extrinsics, dist, errors = calibrate(calib_images, (6, 9), 10)
    print(f"K = {intrinsic}")
    print(f"dist = {dist[0]}")
    print(f"extrinsic matrix for 1st image = {extrinsics[0]}")
