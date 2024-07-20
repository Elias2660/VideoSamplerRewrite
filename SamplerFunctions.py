import logging
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Manager
import time
import random
import torch
import webdataset as wds

import os
from time import sleep 


def sample_video(
    video_path: str,
    sample_list,
    num_samples: int,
    name,
    lock,
    row: pd.Series,
    frames_per_sample: int = 1,
    sample_span=1,
    normalize=True,
    out_channels=1,
    directory_name:str="",
):
    cap = None
    try:
        """
        -return samples given the interval given
        """
        logging.info(f"Sampling {video_path}")
        start_time = time.time()
        begin_frame = row.iloc[2]
        end_frame = row.iloc[3]
        width, height, total_frames = getVideoInfo(video_path)
        available_samples = (end_frame - (sample_span - 1) - begin_frame) // sample_span
        num_samples = min(available_samples, num_samples)

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
        logging.debug(f"Target_Samples: {target_samples}")
        sample_idx = 0
        samples = []
        counts = []
        partial_sample = []

        count = 0
        samples_recorded = False
        frame_of_sample = 0
        logging.info(f"Capture to {video_path} about to be established")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error(f"Failed to open video {video_path}")
            return
        
        if count % 10000 == 0:
            logging.info(f"Frame {count} read from video {video_path}")        
        while count <= end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break

            count += 1
            if count in target_samples:
                logging.debug(
                    f"Frame {count} just triggered the samples_recorded variable"
                )
                samples_recorded = True
                frame_of_sample = 0
                partial_sample = []

            if samples_recorded:
                frame_of_sample += 1
                if normalize:
                    frame = cv2.normalize(
                        frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                    )

                if out_channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                contrast = 1.9  # Simple contrast control [1.0-3.0]
                brightness = 10  # Simple brightness control [0-100]
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


                if out_channels == 1:
                    logging.debug(f"Converting frame {count} to greyscale")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                np_frame = np.array(frame)
                in_frame = torch.tensor(
                    data=np_frame,
                    dtype=torch.uint8,
                ).reshape([1, height, width, out_channels])
                in_frame = in_frame[:, :height, :width, :]

                in_frame = in_frame.permute(0, 3, 1, 2).to(dtype=torch.float)
                logging.debug(f"in_frame shape: {in_frame.shape}")
                logging.debug(f"Tensor has shape {in_frame.shape}")

                partial_sample.append(in_frame)
                counts.append(str(count))

                # read one sample as an image

            if frame_of_sample == frames_per_sample:
                if frames_per_sample == 1:
                    logging.debug(f"Appending partial sample {partial_sample[0]}")
                    t = partial_sample[0]
                    pt_name = f"{directory_name}/{video_path.split('/')[-1]}_{count}.pt"
                    torch.save(t, pt_name)
                    samples.append([pt_name, video_path, counts, row.iloc[1]])

                else:
                    logging.debug(
                        f"Appending partial sample {torch.cat(partial_sample)}"
                    )
                    t = torch.cat(partial_sample)
                    pt_name = f"{directory_name}/{video_path.split('/')[-1]}_{count}.pt"
                    torch.save(t, pt_name)
                    samples.append(
                        [pt_name, video_path, counts, row.iloc[1]]
                    )
                    sample_idx += 1

                frame_of_sample = 0
                sample_idx += 1
                counts = []
                partial_sample = []
                samples_recorded = False
                
        logging.info(
            f"Capture to {video_path} has been released, returning {len(samples)} samples"
        )
        end_time = time.time()
        logging.info("Time taken to sample video: " + str(end_time - start_time))        
        logging.info(f"SAMPLER LIST LENGTH: {len(sample_list)}")
        sample_list.append(samples)

    except Exception as e:
        logging.error(f"Error sampling video {video_path}: {e}")
        raise

    finally:
        logging.info(f"Releasing video capture for {video_path}")
        cap.release()
        sleep(2)
        cv2.destroyAllWindows()
        logging.info(f"Released video capture for {video_path}")
    return

def getVideoInfo(video_path: str):
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
    try:
        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    sleep(2)
    counts = pd.read_csv("counts.csv")
    logging.info(video_path.split("/")[-1])
    logging.info(counts[counts["filename"] == video_path.split("/")[-1]])
    total_frames = counts[counts["filename"] == video_path.split("/")[-1]][
        "framecount"
    ].values[0]


    return width, height, total_frames


# if __name__ == "__main__":
#     with Manager() as manager:
#         lock = manager.Lock()
#         format = "%(asctime)s: %(message)s"
#         logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
#         tar_writer = wds.TarWriter("dataset.tar", encoder=False)
#         sample_video(
#             "2024-07-03 17:20:20.604941.mp4",
#             500,
#             "dataset.tar",
#             lock,
#             pd.Series(
#                 {
#                     "file": "2024-07-03 17:20:20.604941.mp4",
#                     "class": 1,
#                     "begin frame": 0,
#                     "end frame": 1000,
#                 }
#             ),
#             1,
#             1,
#             True,
#             1,
#         )
#         tar_writer.close()
