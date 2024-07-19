import logging
import cv2
import pandas as pd
import numpy as np
import time
import random
import os
import torch
import io
import webdataset as wds
from torchvision import transforms
from WriteToDataset import write_to_dataset


def sample_video(
    video_path,
    num_samples,
    dataset_writer,
    lock,
    row: pd.Series,
    frames_per_sample,
    channels=3,
    begin_frame=None,
    end_frame=None,
    sample_span=1,
    normalize=True,
    out_channels=3,
):
    """
    -return samples given the interval given
    """
    start_time = time.time()

    width, height, total_frames = getVideoInfo(video_path)
    available_samples = (end_frame - (sample_span - 1) - begin_frame) // sample_span
    num_samples = min(available_samples, num_samples)
    if end_frame is None:
        end_frame = total_frames
    if begin_frame is None:
        begin_frame = 0

    logging.info(f"Sampling frames from video {video_path}")
    logging.info(
        f"Total frames: {total_frames}; frame_width: {width}; frame_height: {height}"
    )
    logging.info(f"Sampling {num_samples} samples from {video_path}")
    target_samples = [
        (begin_frame) + x * sample_span
        for x in sorted(
            random.sample(population=range(available_samples), k=num_samples)
        )
    ]
    sample_idx = 0
    samples = []
    counts = []
    partial_sample = []

    count = 0
    sample_recorded = False
    frame_of_sample = 0
    logging.info(f"Capture to {video_path} about to be established")
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count == target_samples[sample_idx]:
            samples_recorded = True
            frame_of_sample = 0
            partial_sample = []

        #  check if sample needed to be read ->
        if samples_recorded:
            # convert to greyscale
            frame_of_sample += 1
            if normalize:
                frame = cv2.normalize(
                    frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )

            if channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            contrast = 1.9  # Simple contrast control [1.0-3.0]
            brightness = 10  # Simple brightness control [0-100]
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

            if channels == 1:
                if counts % 500 == 0:
                    logging.debug(
                        f"Converting frame {count} to gray with shape {frame.shape}/"
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            np_frame = np.array(frame)
            in_frame = torch.tensor(
                data=np_frame,
                dtype=torch.uint8,
            ).reshape([1, height, width, channels])
            in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)

            if counts % 500 == 0:
                logging.debug(
                    f"Appending frame {count} to sample with shape {in_frame.shape}/"
                )
                logging.debug(f"Tensor has shape {in_frame.shape}")

            partial_sample.append(in_frame)
            counts.append(str(count))

        if frame_of_sample == frames_per_sample:
            if frames_per_sample == 1:
                samples.append(partial_sample[0], video_path, counts)

            else:
                samples.append(torch.cat(partial_sample), video_path, counts)
                sample_idx += 1

            frame_of_sample = 0
            sample_idx += 1
            counts = []
            partial_sample = []
            samples_recorded = False

    cap.release()
    logging.info(
        f"Capture to {video_path} has been released, returning {len(samples)} samples"
    )
    end_time = time.time()
    logging.info("Time taken to sample video: " + str(end_time - start_time))

    logging.info(
        "Writing the samples to the dataset ; handing off the the write_to_dataset_function"
    )
    write_to_dataset(
        video_path,
        dataset_writer,
        samples,
        row,
        lock,
        video_path,
        channels,
        frames_per_sample,
        out_channels,
    )


def getVideoInfo(video_path):
    """
    Get the total frames in a video.

    Arguments:
        video_path (str): The path to the video file.
    Returns:
        int: Width
        int: Height
        int: The total number of frames.
    """
    # Following advice from https://kkroening.github.io/ffmpeg-python/index.html
    # First find the size, then set up a stream.
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    counts = pd.read_csv("counts.csv")
    total_frames = counts[counts["filename"] == video_path.split("/")[-1]][
        "framecount"
    ].values[0]
    cap.release()

    return width, height, total_frames
