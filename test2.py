import math
import numpy as np
import cv2
import random
import torch


def vidSamplingCommonCrop(
    height, width, out_height, out_width, scale, x_offset, y_offset
):
    if out_width is None:
        out_width = math.floor(width * scale)
    if out_height is None:
        out_height = math.floor(height * scale)

    crop_x = math.floor((width * scale - out_width) / 2 + x_offset)
    crop_y = math.floor((height * scale - out_height) / 2 + y_offset)

    return out_width, out_height, crop_x, crop_y


def sample_video(
    video_path,
    num_samples,
    frames_per_sample,
    frame_interval,
    out_width=None,
    out_height=None,
    crop_noise=0,
    crop_x_offset=0,
    crop_y_offset=0,
    channels=3,
    begin_frame=None,
    end_frame=None,
    bg_subtract="none",
    normalize=True,
):
    samples = []
    sample_frames_list = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if end_frame is None:
        end_frame = total_frames
    if begin_frame is None:
        begin_frame = 0
    frames = list(range(begin_frame, end_frame, frame_interval + 1))
    frames = frames[:num_samples]
    
    print(f"Sampling {len(frames)} frames from {video_path}")
    print(f"Total frames: {total_frames}; frame_width: {frame_width}; frame_height: {frame_height}")
    


    count = 0
    for frame_no in frames:
        print(count, end="\r")
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_no}")
            continue

        print(frame.shape)
        
        if channels == 1:
            norm_frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # norm_frame = cv2.createBackgroundSubtractorMOG2().apply(norm_frame)
            norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)
            norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)
            
            alpha = 1.9  # Simple contrast control [1.0-3.0]
            beta = 10   # Simple brightness control [0-100]
            new_image = cv2.convertScaleAbs(norm_frame, alpha=alpha, beta=beta)
            
            cv2.imwrite(f"(new_image){count}.png", new_image)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            print(new_image.shape)
            
            norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)
            np_frame = np.array(norm_frame)
            print(np_frame.shape)
            in_frame = torch.tensor(data=np_frame, dtype=torch.uint8,
                                ).reshape([1, frame_height, frame_width, channels])
            print(in_frame)
            print(in_frame.shape)
            in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)
            print(in_frame)
            

        samples.append(frame)
        sample_frames_list.append(frame_no)
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    return samples, sample_frames_list


# Example usage
video_path = "2024-07-03 17:20:20.604941.mp4"
num_samples = 1
frames_per_sample = 1
frame_interval = 0

samples, sample_frames = sample_video(
    video_path,
    num_samples,
    frames_per_sample,
    frame_interval,
    out_width=960,
    out_height=720,
    scale=1.0,
    crop_x_offset=0,
    crop_y_offset=0,
    channels=1,
    begin_frame=0,
    end_frame=20000,
    normalize=False,
    bg_subtract="knn"
)

# samples is a list of processed frames
# sample_frames is a list of frame numbers
