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
    lock,
):
    start_time = time.time()
    cap = None
    count = 0
    try:
        dataframe = old_df.copy(deep=True)
        dataframe.reset_index(drop=True, inplace=True)
        t_s = []

        logging.debug(f"Dataframe for {video} about to be prepared (0)")
        width, height = getVideoInfo(video)
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
                f"Target samples for {video}: {target_samples[0]} begin, {target_samples[-1]} end"
            )
            logging.debug(f"Target samples for {video}: {target_samples}")
            t_s.append(target_samples)

        logging.info(f"Size of target sample list for {video}: {len(t_s)}")
        logging.debug(f"Dataframe for {video} about to be prepared(1)")

        dataframe["counts"] = ""
        dataframe["counts"] = dataframe["counts"].apply(list)
        dataframe["partial_sample"] = ""
        dataframe["partial_sample"] = dataframe["partial_sample"].apply(list)
        dataframe["samples_recorded"] = False
        dataframe["frame_of_sample"] = 0

        logging.debug(dataframe.head())

        logging.info(f"Capture to {video} about to be established")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logging.error(f"Failed to open video {video}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            logging.debug(f"Frame {count} read from video {video}")
            if count % 10000 == 0:
                logging.info(f"Frame {count} read from video {video}")
            for index, row in dataframe.iterrows():
                if t_s[index][0] > count or t_s[index][-1] < count:
                    continue
                logging.debug(
                    f"length of target sample sample list: {len(t_s)} \n index: {index}"
                )
                if count in t_s[index]:
                    logging.debug(f"Frame {count} triggered samples_recorded")
                    dataframe.at[index, "samples_recorded"] = True
                    dataframe.at[index, "frame_of_sample"] = 0
                    dataframe.at[index, "partial_sample"] = []

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
                    logging.debug(f"in_frame shape: {in_frame.shape}")
                    logging.debug(f"Tensor has shape {in_frame.shape}")
                    dataframe.at[index, "partial_sample"].append(in_frame)
                    dataframe.at[index, "counts"].append(str(count))
                    # read one sample as an image
                    if row["frame_of_sample"] == frames_per_sample:
                        logging.debug(f"Saving sample at frame {count} for {video}")
                        save_sample(
                            row, video, frames_per_sample, dataframe, index, lock
                        )

                        logging.info(f"Saved sample at frame {count} for {video}")
                        dataframe.at[index, "frame_of_sample"] = 0
                        dataframe.at[index, "counts"] = []
                        dataframe.at[index, "partial_sample"] = []
                        dataframe.at[index, "samples_recorded"] = False

        logging.info(f"Capture to {video} has been released, writing samples")
        end_time = time.time()
        logging.info("Time taken to sample video: " + str(end_time - start_time))

    except Exception as e:
        logging.error(f"Error sampling video {video}: {e}")
        raise

    finally:
        logging.info(f"Releasing video capture for {video}")
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Released video capture for {video}")
    return


def save_sample(row, video, frames_per_sample, dataframe, index, lock):
    try:
        directory_name = row.loc["data_file"].replace(".csv", "") + "_samplestemporary"
        s_c = "-".join([str(x) for x in row["counts"]])
        d_name = row.iloc[1]

        # with lock:
        if frames_per_sample == 1:
            t = dataframe.loc[index, "partial_sample"][0]
            pt_name = (
                f"{directory_name}/{video.replace(' ', 'SPACE')}_{d_name}.pt".replace(
                    "\x00", ""
                )
            )
            s_c_file = open(
                f"{directory_name}txt/{video.replace(' ', 'SPACE')}_{d_name}.txt".replace(
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
            t = torch.cat(dataframe.at[index, "partial_sample"])
            pt_name = (
                f"{directory_name}/{video.replace(' ', 'SPACE')}_{d_name}.pt".replace(
                    "\x00", ""
                )
            )
            s_c_file = open(
                f"{directory_name}txt/{video.replace(' ', 'SPACE')}_{d_name}.txt".replace(
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
            f"Saved sample {s_c} for {video}, with name {directory_name}/{pt_name}"
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
    bg_subtract,
):
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
    Get the total frames in a video.

    Arguments:
        video (str): The path to the video file.
    Returns:
        int: Width
        int: Height
        int: The total number of frames.
    """
    # Following advice from https://kkroening.github.io/ffmpeg-python/index.html
    # First find the size, then set up a stream.
    try:
        cap = cv2.VideoCapture(video)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()

    return width, height
