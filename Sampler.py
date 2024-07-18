import argparse
import cv2
import numpy as np
import pandas as pd
import os
import random
import pandas
import logging
import math


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
    total_frames = counts[counts["filename"] == video_path.split("/")[-1]]["count"][
        "framecount"
    ]
    cap.release()

    return width, height, total_frames


def vidSamplingCommonCrop(
    height, width, out_height, out_width, scale, x_offset, y_offset
):
    """
    Return the common cropping parameters used in dataprep and annotations.

    Arguments:
        height     (int): Height of the video
        width      (int): Width of the video
        out_height (int): Height of the output patch
        out_width  (int): Width of the output patch
        scale    (float): Scale applied to the original video
        x_offset   (int): x offset of the crop (after scaling)
        y_offset   (int): y offset of the crop (after scaling)
    Returns:
        out_width, out_height, crop_x, crop_y
    """

    if out_width is None:
        out_width = math.floor(width * scale)
    if out_height is None:
        out_height = math.floor(height * scale)

    crop_x = math.floor((width * scale - out_width) / 2 + x_offset)
    crop_y = math.floor((height * scale - out_height) / 2 + y_offset)

    return out_width, out_height, crop_x, crop_y


def get_samples(
    video_path,
    num_samples,
    frames_per_sample,
    frame_interval,
    out_width=None,
    out_height=None,
    crop_noise=0,
    scale=1.0,
    crop_x_offset=0,
    crop_y_offset=0,
    channels=3,
    begin_frame=None,
    end_frame=None,
    bg_subtract="none",
    normalize=True,
):
    """
    Samples have no overlaps. For example, a 10 second video at 30fps has 300 samples of 1
    frame, 150 samples of 2 frames with a frame interval of 0, or 100 samples of 2 frames with a
    frame interval of 1.
    Arguments:
        video_path  (str): Path to the video.
        num_samples (int): Number of samples yielded from VideoSampler's iterator.
        frames_per_sample (int):  Number of frames in each sample.
        frame_interval    (int): Number of frames to skip between each sampled frame.
        out_width     (int): Width of output images, or the original width if None.
        out_height    (int): Height of output images, or the original height if None.
        crop_noise    (int): Noise to add to the crop location (in both x and y dimensions)
        scale       (float): Scale factor of each dimension
        crop_x_offset (int): x offset of crop, in pixels, from the original image
        crop_y_offset (int): y offset of crop, in pixels, from the original image
        channels      (int): Numbers of channels (3 for RGB or 1 luminance/Y/grayscale/whatever)
        begin_frame   (int): First frame to possibly sample.
        end_frame     (int): Final frame to possibly sample.
        bg_subtract   (str): Type of background subtraction to use (mog2 or knn), or none.
        normalize    (bool): True to normalize image channels (done independently)
    """
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("Gathering Environment Variables")
    bg_subtractor = None
    if "none" != bg_subtract:
        from cv2 import createBackgroundSubtractorMOG2, createBackgroundSubtractorKNN

        if "mog2" == bg_subtract:
            bg_subtractor = createBackgroundSubtractorMOG2()
        elif "knn" == bg_subtract:
            bg_subtractor = createBackgroundSubtractorKNN()

    width, height, total_frames = getVideoInfo(video_path)

    if out_width is None or out_height is None:
        crop_noise = 0
    else:
        crop_noise = crop_noise
        
    if begin_frame is None:
        begin_frame = 1
    else:
        begin_frame = int(begin_frame)
        
    if end_frame is None:
        end_frame = total_frames
    else:
        # Don't attempt to sample more frames than there exist.
        end_frame = min(int(end_frame), total_frames)
        
    sample_span = frames_per_sample + (frames_per_sample - 1) * frame_interval
    available_samples = (end_frame - (sample_span - 1) - begin_frame)//sample_span
    num_samples = min(available_samples, num_samples)
    logging.info(f"Video begin and end frames are {begin_frame} and {end_frame}")
    logging.info(f"Video has {available_samples} available samples of size {sample_span} and {num_samples} will be sampled")
    
    