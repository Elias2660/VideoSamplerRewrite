# TODO: ADD SCALING?

"""
SamplerFunctions.py

This module contains functions for sampling frames from videos and processing them for dataset preparation.

Functions:
    sample_video(video: str, old_df: pd.DataFrame, number_of_samples_max: int, frames_per_sample: int, normalize: bool, out_channels: int, sample_span: int):
        Samples frames from a video based on the provided parameters, writing the samples to folders.

    save_sample(row, partial_frames, video, frames_per_sample, count, spc):
        Saves the sampled frames to disk in the specified format.

    apply_video_transformations(frame, count, normalize, out_channels, height, width):
        Applies transformations to the video frames such as normalization.

    getVideoInfo(video: str):
        Retrieves information about the video such as frame count, width, and height.

Constants:
    target_sample_list: List of target samples for each frame.
    target_samples: List of samples to be targeted.
"""

import logging
import cv2
import pandas as pd
import numpy as np
import os
import datetime
import time
import math
import random
import torch
from concurrent.futures import ThreadPoolExecutor


def sample_video(
    video: str,
    old_df: pd.DataFrame,
    number_of_samples_max: int,
    frames_per_sample: int,
    normalize: bool,
    out_channels: int,
    sample_span: int,
    out_height: int = None,
    out_width: int = None,
    x_offset: int = 0,
    y_offset: int = 0,
    crop: bool = False,
):
    """
    Samples frames from a video based on the provided parameters, writing the samples to folders

    Args:
        video (str): The path to the video file.
        old_df (pd.DataFrame): The original DataFrame containing information about the video frames.
        number_of_samples_max (int): The maximum number of samples to be taken from the video.
        frames_per_sample (int): The number of frames to be included in each sample.
        normalize (bool): Flag indicating whether to normalize the sampled frames.
        out_channels (int): The number of output channels for the sampled frames.
        sample_span (int): The span between each sample.

    Returns:
        None
    """
    start_time = (
        time.time()
    )  # start the timer to determine how long it takes to sample the video
    cap = None
    count = 0
    sample_count = 0
    try:
        dataframe = old_df.copy(deep=True)
        dataframe.reset_index(drop=True, inplace=True)
        target_sample_list = (
            []
        )  # list of lists, these don't work well the the dataframe
        partial_frame_list = []

        logging.debug(f"Dataframe for {video} about to be prepared (0)")
        width, height = getVideoInfo(video)

        # Extract necessary columns
        begin_frames = dataframe.iloc[:, 2].values
        end_frames = dataframe.iloc[:, 3].values

        # Calculate available samples
        available_samples = (
            end_frames - (sample_span - frames_per_sample) - begin_frames
        ) // sample_span

        # Determine the number of samples
        num_samples = np.minimum(available_samples, number_of_samples_max)

        # Generate target samples
        target_samples_list = [
            sorted(random.sample(range(avail), num)) if avail > 0 else []
            for avail, num in zip(available_samples, num_samples)
        ]

        # Adjust target samples to start from begin_frame
        target_samples_list = [
            [begin_frame + x * sample_span for x in target_samples]
            for begin_frame, target_samples in zip(begin_frames, target_samples_list)
        ]

        # Log and append results
        for target_samples in target_samples_list:
            if target_samples:
                logging.debug(
                    f"Target samples for {video}: {target_samples[0]} begin, {target_samples[-1]} end, number of samples {len(target_samples)}, frames per sample: {frames_per_sample}"
                )
                logging.debug(f"Target samples for {video}: {target_samples}")
            target_sample_list.append(target_samples)
            partial_frame_list.append([])

        logging.debug(
            f"Size of target sample list for {video}: {len(target_sample_list)}"
        )
        logging.debug(f"Dataframe for {video} about to be prepared(1)")

        dataframe["counts"] = ""
        dataframe["counts"] = dataframe["counts"].apply(list)
        dataframe["samples_recorded"] = False
        dataframe["frame_of_sample"] = 0

        logging.debug(dataframe.head())

        logging.info(f"Capture to {video} about to be established")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logging.error(f"Failed to open video {video}")
            return
        with ThreadPoolExecutor(max_workers=6) as executor:
            while True:
                ret, frame = cap.read()  # read a frame
                if not ret:
                    break
                count += 1  # count the frame
                logging.debug(f"Frame {count} read from video {video}")
                if count % 10000 == 0:
                    logging.info(f"Frame {count} read from video {video}")
                spc = 0

                relevant_rows = dataframe[
                    (
                        dataframe.index.map(
                            lambda idx: target_sample_list[idx][0]
                            <= count
                            <= target_sample_list[idx][-1]
                        )
                    )
                ]

                for index, row in relevant_rows.iterrows():
                    if (
                        target_sample_list[index][0] > count
                        or target_sample_list[index][-1] < count
                    ):
                        # skip if the frame is not in the target sample list
                        continue
                    logging.debug(
                        f"length of target sample sample list: {len(target_sample_list)} \n index: {index}"
                    )
                    if count in target_sample_list[index]:
                        # start recoding samples
                        logging.debug(f"Frame {count} triggered samples_recorded")
                        dataframe.at[index, "samples_recorded"] = True

                    if row["samples_recorded"]:

                        dataframe.at[index, "frame_of_sample"] += 1
                        in_frame = apply_video_transformations(
                            frame,
                            count,
                            normalize,
                            out_channels,
                            height,
                            width,
                            crop,
                            x_offset,
                            y_offset,
                            out_width,
                            out_height,
                        )
                        partial_frame_list[index].append(in_frame)
                        dataframe.at[index, "counts"].append(str(count))

                        if (
                            int(row["frame_of_sample"]) == int(frames_per_sample) - 1
                        ):  # -1 because we start at 0
                            spc += 1  # scramble to make sure every saved .pt sample is unique
                            executor.submit(
                                save_sample,
                                row,
                                partial_frame_list[index],
                                video,
                                frames_per_sample,
                                count,
                                spc,
                            )
                            if sample_count % 100 == 0:
                                logging.info(
                                    f"Saved sample {sample_count} at frame {count} for {video}"
                                )

                            sample_count += 1
                            # reset the dataframe row
                            dataframe.at[index, "frame_of_sample"] = 0
                            dataframe.at[index, "counts"] = []
                            partial_frame_list[index] = []
                            dataframe.at[index, "samples_recorded"] = False
                            
                
            logging.info(f"Capture to {video} has been released, writing samples")
            end_time = time.time()
            logging.info("Time taken to sample video: " + str(datetime.timedelta(seconds=(end_time - start_time))))
            executor.shutdown(wait=True)
    except Exception as e:
        logging.error(f"Error sampling video {video}: {e}")
        raise

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Released video capture for {video}")
    return


import os
import torch
import logging


def save_sample(row, partial_frames, video, frames_per_sample, count, spc):
    """
    Save a sample of frames to disk.

    Args:
        row (pandas.Series): The row containing information about the sample.
        partial_frames (list): List of partial frames to be saved.
        video (str): The name of the video.
        frames_per_sample (int): The number of frames per sample.
        count (int): The count of the sample.
        spc (int): The spc of the sample.

    Raises:
        Exception: If there is an error saving the sample.

    Returns:
        None
    """
    try:
        directory_name = row.loc["data_file"].replace(".csv", "") + "_samplestemporary"
        s_c = "-".join([str(x) for x in row["counts"]])
        d_name = row.iloc[1]
        video_name = video.replace(" ", "SPACE")
        base_name = f"{directory_name}/{video_name}_{d_name}_{count}_{spc}".replace(
            "\x00", ""
        )
        pt_name = f"{base_name}.pt"
        txt_name = (
            f"{directory_name}txt/{video_name}_{d_name}_{count}_{spc}.txt".replace(
                "\x00", ""
            )
        )

        # Save the sample counts to a text file
        with open(txt_name, "w+") as s_c_file:
            s_c_file.write(s_c)

        # Check for overwriting
        if pt_name in os.listdir(directory_name):
            logging.error(f"Overwriting {pt_name}")

        # Save the tensor
        if frames_per_sample == 1:
            t = partial_frames[0]
        else:
            t = torch.cat(partial_frames)

        torch.save(t, pt_name)
        logging.debug(f"Saved sample {s_c} for {video}, with name {pt_name}")

    except Exception as e:
        logging.error(f"Error saving sample: {e}")
        raise


def apply_video_transformations(
    frame,
    count: int,
    normalize: bool,
    out_channels: int,
    height: int,
    width: int,
    crop: bool = False,
    x_offset: int = 0,
    y_offset: int = 0,
    out_width: int = 400,
    out_height: int = 400,
):
    """
    Apply transformations to a video frame.

    Args:
        frame: The input video frame.
        count (int): The frame count.
        normalize (bool): Flag indicating whether to normalize the frame.
        out_channels (int): The number of output channels.
        height (int): The desired height of the frame.
        width (int): The desired width of the frame.

    Returns:
        torch.Tensor: The transformed video frame as a tensor.
    """
    if normalize:
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if out_channels == 1:
        logging.debug(f"Converting frame {count} to greyscale since out_channels is 1")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    logging.debug(f"Frame shape: {frame.shape}, adding contrast to partial sample")
    contrast = 1.9  # Simple contrast control [1.0-3.0]
    brightness = 10  # Simple brightness control [0-100]
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    logging.debug(f"Frame shape: {frame.shape}, converting to a tensor")
    np_frame = np.array(frame)

    in_frame = (
        torch.tensor(
            data=np_frame,
            dtype=torch.float32,
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
    )  # Shape: [1, C, H, W]

    if crop:
        out_width, out_height, crop_x, crop_y = vidSamplingCommonCrop(
            height, width, out_height, out_width, 1, x_offset, y_offset
        )
        in_frame = in_frame[
            :, :, crop_y : crop_y + out_height, crop_x : crop_x + out_width
        ]

    return in_frame


def getVideoInfo(video: str):
    """
    Retrieves the width and height of a video.

    Parameters:
    video (str): The path to the video file.

    Returns:
    tuple: A tuple containing the width and height of the video.
    """

    try:
        cap = cv2.VideoCapture(video)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()

    return width, height


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
