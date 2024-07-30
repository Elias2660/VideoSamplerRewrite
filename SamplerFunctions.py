import logging
import cv2
import pandas as pd
import numpy as np
import os
import time
import random
import torch


def sample_video(
    video: str,
    old_df: pd.DataFrame,
    number_of_samples_max: int,
    frames_per_sample: int,
    normalize: bool,
    out_channels: int,
    bg_subtract,
    sample_span: int,
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
        bg_subtract: Background subtraction method to be applied to the frames.
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

        # 49-81 setups up the dataset
        for index, row in dataframe.iterrows():
            begin_frame = row.iloc[2]
            end_frame = row.iloc[3]
            available_samples = (
                end_frame - (sample_span - frames_per_sample) - begin_frame
            ) // sample_span
            num_samples = min(available_samples, number_of_samples_max)

            target_samples = [
                (begin_frame) + x * sample_span
                for x in sorted(
                    random.sample(population=range(available_samples), k=num_samples)
                )
            ]
            logging.info(
                f"Target samples for {video}: {target_samples[0]} begin, {target_samples[-1]} end, frames per sample: {frames_per_sample}"
            )
            logging.debug(f"Target samples for {video}: {target_samples}")
            target_sample_list.append(target_samples)
            partial_frame_list.append([])

        logging.info(
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

        while True:
            ret, frame = cap.read()  # read a frame
            if not ret:
                break
            count += 1  # count the frame
            logging.debug(f"Frame {count} read from video {video}")
            if count % 10000 == 0:
                logging.info(f"Frame {count} read from video {video}")
            spc = 0
            for index, row in dataframe.iterrows():
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
                        bg_subtract=bg_subtract,
                    )
                    partial_frame_list[index].append(in_frame)
                    dataframe.at[index, "counts"].append(str(count))

                    if (
                        int(row["frame_of_sample"]) == int(frames_per_sample) - 1
                    ):  # -1 because we start at 0
                        spc += (
                            1  # scramble to make sure every saved .pt sample is unique
                        )

                        save_sample(
                            row,
                            partial_frame_list[index],
                            video,
                            frames_per_sample,
                            count,
                            spc,
                        )
                        if sample_count % 100 == 0:
                            logging.info(f"Saved sample at frame {count} for {video}")

                        sample_count += 1
                        # reset the dataframe row
                        dataframe.at[index, "frame_of_sample"] = 0
                        dataframe.at[index, "counts"] = []
                        partial_frame_list[index] = []
                        dataframe.at[index, "samples_recorded"] = False

        logging.info(f"Capture to {video} has been released, writing samples")
        end_time = time.time()
        logging.info("Time taken to sample video: " + str(end_time - start_time))

    except Exception as e:
        logging.error(f"Error sampling video {video}: {e}")
        raise

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Released video capture for {video}")
    return


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

        if frames_per_sample == 1:
            t = partial_frames[0]
            pt_name = f"{directory_name}/{video.replace(' ', 'SPACE')}_{d_name}_{count}_{spc}.pt".replace(
                "\x00", ""
            )
            s_c_file = open(
                f"{directory_name}txt/{video.replace(' ', 'SPACE')}_{d_name}_{count}_{spc}.txt".replace(
                    "\x00", ""
                ),
                "w+",
            )
            s_c_file.write(s_c)
            s_c_file.close()
            # check for overwriting
            if pt_name in os.listdir(directory_name):
                logging.error(f"Overwriting {pt_name}")
            torch.save(t, pt_name)
        else:
            t = torch.cat(partial_frames)
            pt_name = f"{directory_name}/{video.replace(' ', 'SPACE')}_{d_name}_{count}_{spc}.pt".replace(
                "\x00", ""
            )
            s_c_file = open(
                f"{directory_name}txt/{video.replace(' ', 'SPACE')}_{d_name}_{count}_{spc}.txt".replace(
                    "\x00", ""
                ),
                "w+",
            )
            s_c_file.write(s_c)
            s_c_file.close()
            if pt_name in os.listdir(directory_name):
                logging.error(f"Overwriting {pt_name}")
            torch.save(t, pt_name)
        logging.info(
            f"Saved sample {s_c} for {video}, with name {pt_name}"
        )
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
    bg_subtract,  # TODO  Background subtraction parameter,  not used RM
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
        bg_subtract: Background subtraction parameter.

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

    if out_channels == 1:
        logging.debug(f"Converting frame {count} to greyscale")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    logging.debug(f"Frame shape: {frame.shape}, converting to a tensor")
    np_frame = np.array(frame)
    in_frame = torch.tensor(
        data=np_frame,
        dtype=torch.uint8,
    ).reshape([1, height, width, out_channels])

    in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)
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
