# TODO: ADD SCALING?
"""
SamplerFunctions.py

This module contains functions for sampling frames from videos and processing them for dataset preparation.

Functions:
    sample_video(
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
        max_batch_size: int = 10,
    ):
        Samples frames from a video based on the provided parameters, writing the samples to folders.

    save_sample(batch):
        Saves the sampled frames to disk in the specified format.

    apply_video_transformations(frame, count, normalize, out_channels, height, width):
        Applies transformations to the video frames such as normalization.

    getVideoInfo(video: str):
        Retrieves information about the video such as frame count, width, and height.

Constants:
    target_sample_list: List of target samples for each frame.
    target_samples: List of samples to be targeted.
"""
import datetime
import gc
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch


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
    max_batch_size: int = 50,
    max_threads_pic_saving: int = 10,
):
    """Samples frames from a video based on the provided parameters, writing the samples to folders

    :param video: The path to the video file.
    :type video: str
    :param old_df: The original DataFrame containing information about the video frames.
    :type old_df: pd.DataFrame
    :param number_of_samples_max: The maximum number of samples to be taken from the video.
    :type number_of_samples_max: int
    :param frames_per_sample: The number of frames to be included in each sample.
    :type frames_per_sample: int
    :param normalize: Flag indicating whether to normalize the sampled frames.
    :type normalize: bool
    :param out_channels: The number of output channels for the sampled frames.
    :type out_channels: int
    :param sample_span: The span between each sample.
    :type sample_span: int

    :returns: None

    """
    start_time = (
        time.time()
    )  # start the timer to determine how long it takes to sample the video
    logging.info(f"Capture to {video} about to be established")

    cap = None
    count = 0
    sample_count = 0
    try:
        dataframe = old_df.copy(deep=True)
        dataframe.reset_index(drop=True, inplace=True)
        target_sample_list = (
            [])  # list of lists, these don't work well the the dataframe
        partial_frame_list = []

        logging.debug(f"Dataframe for {video} about to be prepared (0)")
        width, height = getVideoInfo(video)

        # Extract necessary columns
        begin_frames = dataframe.iloc[:, 2].values
        end_frames = dataframe.iloc[:, 3].values


        # Calculate available samples for each row in the dataframe
        available_samples = (end_frames - (sample_span - frames_per_sample) - begin_frames) // sample_span

        # Generate target samples in one comprehension
        target_samples_list = [
            [] if avail <= 0 else [
                begin_frame + s * sample_span
                for s in sorted(
                    np.random.choice(
                        range(avail),
                        size=min(avail, number_of_samples_max),
                        replace=False
                    )
                )
            ]
            for begin_frame, avail in zip(begin_frames, available_samples)
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

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logging.error(f"Failed to open video {video}")
            return
        with ThreadPoolExecutor(
                max_workers=max_threads_pic_saving) as executor:
            batch = []  # using batching to optimize treading
            while True:
                ret, frame = cap.read()  # read a frame
                if not ret:
                    break
                count += 1  # count the frame
                if count % 10000 == 0 and count != 0:
                    logging.debug(f"Frame {count} read from video {video}")
                spc = 0

                relevant_rows = dataframe[(
                    dataframe.index.map(lambda idx: target_sample_list[idx][
                        0] <= count <= target_sample_list[idx][-1]))]

                for index, row in relevant_rows.iterrows():
                    if (target_sample_list[index][0] > count
                            or target_sample_list[index][-1] < count):
                        # skip if the frame is not in the target sample list
                        continue
                    logging.debug(
                        f"length of target sample sample list: {len(target_sample_list)} \n index: {index}"
                    )
                    if count in target_sample_list[index]:
                        # start recoding samples
                        logging.debug(
                            f"Frame {count} triggered samples_recorded")
                        dataframe.at[index, "samples_recorded"] = True

                    if dataframe.at[index, "samples_recorded"]:

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

                        if (int(row["frame_of_sample"]) ==
                                int(frames_per_sample) -
                                1):  # -1 because we start at 0
                            # scramble to make sure every saved .npz sample is unique
                            spc += 1
                            batch.append([
                                row,
                                partial_frame_list[index],
                                video,
                                frames_per_sample,
                                count,
                                spc,
                            ])
                            if len(batch) >= max_batch_size:
                                executor.submit(
                                    save_sample,
                                    batch,
                                )
                                batch = []  # reset the batch
                                # don't know if completely necessary, but was facing
                                # odd memory issues earlier
                                gc.collect()
                            if sample_count % 10000 == 0 and sample_count != 0:
                                logging.info(
                                    f"Saved sample {sample_count} at frame {count} for {video}"
                                )

                            sample_count += 1
                            # reset the dataframe row
                            dataframe.at[index, "frame_of_sample"] = 0
                            dataframe.at[index, "counts"] = []
                            partial_frame_list[index] = []
                            dataframe.at[index, "samples_recorded"] = False

            if len(batch) > 0:
                save_sample(batch)

        executor.shutdown(wait=True)
        end_time = time.time()
        logging.info(  # log the time taken to sample the video
            f"Time taken to sample video {video}: {str(datetime.timedelta(seconds=(end_time - start_time)))}"
            f" wrote {sample_count} samples, {str(datetime.timedelta(seconds=((end_time - start_time)/sample_count)))} per sample"
        )
    except Exception as e:
        logging.error(f"Error sampling video {video}: {e}")
        executor.shutdown(wait=False)  # the threads are shut down if error
        raise e

    finally:
        cap.release()
        cv2.destroyAllWindows()
        gc.collect()
    return


def save_sample(batch):
    # batch is a table with: row, partial_frames, video, frames_per_sample, count, spc
    """Save a sample of frames to disk (per‐sample subdirectories inside your two temp dirs)."""

    for row, partial_frames, video, fps, count, spc in batch:
        base = row.loc["data_file"].replace(".csv", "")
        png_root = f"{base}_samplestemporary"
        txt_root = f"{base}_samplestemporarytxt"
        os.makedirs(png_root, exist_ok=True)
        os.makedirs(txt_root, exist_ok=True)

        vid = video.replace(" ", "SPACE")
        cls = row.iloc[1]
        key = f"{vid}_{cls}_{count}_{spc}"

        # write counts
        txt_path = os.path.join(txt_root, f"{key}.txt")
        with open(txt_path, "w") as f:
            f.write("-".join(str(x) for x in row["counts"]))

        # write frames under their own subfolder
        sample_dir = os.path.join(png_root, key)
        os.makedirs(sample_dir, exist_ok=True)
        
        for i, frame_tensor in enumerate(partial_frames):
            # Handle grayscale correctly - extract the single channel data
            # The tensor shape should be [1, 1, height, width]
            arr = frame_tensor.squeeze(0).squeeze(0).cpu().numpy().clip(0, 255).astype(np.uint8)
            
            # Keep the original filename format
            frame_path = os.path.join(sample_dir, f"frame_{i:03d}.png")
            
            # Save directly as grayscale
            cv2.imwrite(frame_path, arr)

        logging.debug(f"Saved sample {key}: frames→{sample_dir}, txt→{txt_path}")




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
    """Apply transformations to a video frame.

    :param frame: The input video frame.
    :param count: The frame count.
    :type count: int
    :param normalize: Flag indicating whether to normalize the frame.
    :type normalize: bool
    :param out_channels: The number of output channels.
    :type out_channels: int
    :param height: The desired height of the frame.
    :type height: int
    :param width: The desired width of the frame.
    :type width: int
    :param crop: bool:  (Default value = False)
    :param x_offset: int:  (Default value = 0)
    :param y_offset: int:  (Default value = 0)
    :param out_width: int:  (Default value = 400)
    :param out_height: int:  (Default value = 400)

    """
    # history: pulled, with minimal edits, from the code from bee_analysis
    if normalize:
        frame = cv2.normalize(frame,
                              None,
                              alpha=0,
                              beta=255,
                              norm_type=cv2.NORM_MINMAX)

    # Apply contrast and brightness adjustments
    contrast = 1.9  # Simple contrast control [1.0-3.0]
    brightness = 10  # Simple brightness control [0-100]
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    if out_channels == 1:
        # Convert to grayscale BUT DON'T CONVERT BACK TO BGR
        logging.debug(
            f"Converting frame {count} to true grayscale (1-channel)")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create tensor differently for grayscale - don't permute dimensions
        np_frame = np.array(frame)
        in_frame = torch.tensor(
            data=np_frame,
            dtype=torch.float16
        ).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    else:
        # Standard RGB processing
        logging.debug(f"Keeping frame {count} as RGB (3-channel)")
        np_frame = np.array(frame)
        in_frame = torch.tensor(
            data=np_frame,
            dtype=torch.float16
        ).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]

    if crop:
        out_width, out_height, crop_x, crop_y = vidSamplingCommonCrop(
            height, width, out_height, out_width, 1, x_offset, y_offset)
        in_frame = in_frame[:, :, crop_y:crop_y + out_height,
                          crop_x:crop_x + out_width]

    return in_frame


def getVideoInfo(video: str):
    """Retrieves the width and height of a video.

    :param video: str
    :param video: str:
    :param video: str:
    :param video: str:
    :param video: str:
    :returns: tuple: A tuple containing the width and height of the video.

    """

    try:
        cap = cv2.VideoCapture(video)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()

    return width, height


def vidSamplingCommonCrop(height, width, out_height, out_width, scale,
                          x_offset, y_offset):
    """Return the common cropping parameters used in dataprep and annotations.

    :param height: int
    :param width: int
    :param out_height: int
    :param out_width: int
    :param scale: float
    :param x_offset: int
    :param y_offset: int
    :returns: out_width, out_height, crop_x, crop_y

    """

    if out_width is None:
        out_width = math.floor(width * scale)
    if out_height is None:
        out_height = math.floor(height * scale)

    crop_x = math.floor((width * scale - out_width) / 2 + x_offset)
    crop_y = math.floor((height * scale - out_height) / 2 + y_offset)

    return out_width, out_height, crop_x, crop_y