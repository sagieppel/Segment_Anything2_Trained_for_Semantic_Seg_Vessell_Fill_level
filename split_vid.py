import cv2
import numpy as np


def split_video(input_path, output_prefix):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frames per segment
    frames_per_segment = total_frames // 4

    # Define codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_files = [cv2.VideoWriter(f'{output_prefix}{i + 1}.mp4', fourcc, fps, (width, height)) for i in range(4)]

    # Read and write frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        segment = i // frames_per_segment
        if segment < 4:  # Ensure we don't go out of bounds
            out_files[segment].write(frame)

    # Release everything
    cap.release()
    for out in out_files:
        out.release()

    cv2.destroyAllWindows()


# Usage
input_video = 'input_video.mp4'  # Replace with your input video path
output_prefix = ''  # This will create 1.mp4, 2.mp4, etc. in the current directory
split_video(input_video, output_prefix)