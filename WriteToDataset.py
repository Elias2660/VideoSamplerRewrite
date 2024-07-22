import webdataset as wds
import pandas as pd
import os
import logging
import time
import multiprocessing
import torch
import subprocess
from torchvision import transforms
import cv2
import io
import time
import re


def write_to_dataset(
    directory: str,
    tar_file: str,
    frames_per_sample: int = 1,
    out_channels: int = 1,
):
    try:
        tar_writer = wds.TarWriter(tar_file, encoder=False)
        start_time = time.time()

        logging.info(f"Reading in the samples from {directory}")
        command = f"ls {directory}"
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        file_list = sorted(
            [ansi_escape.sub("", line) for line in result.stdout.splitlines()]
        )

        for file in file_list:
            s = file.split(".")[0].split("/")[-1].split("_")
            filename, d_name, frame_num = s
            video_path = filename
            sample_class = d_name
            frame = torch.load(filename)
            logging.debug(f"Writing sample to dataset")
            logging.debug(f"Frame shape: {frame.shape}")
            logging.debug(f"Frame number: {frame_num}")
            logging.debug(f"Video path: {video_path}")
            logging.debug(f"frame type: {type(frame)}")

            base_name = os.path.basename(video_path).replace(" ", "_").replace(".", "_")
            video_time = os.path.basename(video_path).split(".")[0]
            time_sec = time.mktime(time.strptime(video_time, "%Y-%m-%d %H:%M:%S"))
            time_struct = time.localtime(time_sec + int(frame_num[0]) // 3)
            curtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
            metadata = f"{video_path},{frame_num[0]},{curtime}"

            if 1 == frames_per_sample:
                if 3 == out_channels:
                    img = transforms.ToPILImage()(frame[0] / 255.0).convert("RGB")
                else:
                    img = transforms.ToPILImage()(frame[0] / 255.0).convert("L")

                with img:
                    buf = io.BytesIO()
                    img.save(fp=buf, format="png")
                sample = {
                    "__key__": "_".join((base_name, "_".join(frame_num))),
                    "0.png": buf.getbuffer(),
                    "cls": str(sample_class).encode("utf-8"),
                    "metadata.txt": metadata.encode("utf-8"),
                }
            else:
                # Save multiple pngs
                buffers = []

                for i in range(frames_per_sample):
                    if 3 == out_channels:
                        img = transforms.ToPILImage()(frame[i] / 255.0).convert("RGB")
                    else:
                        img = transforms.ToPILImage()(frame[i] / 255.0).convert("L")
                        with img:
                            buffers.append(io.BytesIO())
                            img.save(fp=buffers[-1], format="png")

                    sample = {
                        "__key__": "_".join((base_name, "_".join(frame_num))),
                        "cls": str(sample_class).encode("utf-8"),
                        "metadata.txt": metadata.encode("utf-8"),
                    }
                    for i in range(frames_per_sample):
                        sample[f"{i}.png"] = buffers[i].getbuffer()

            logging.info(f"Writing sample to dataset tar file")
            tar_writer.write(sample)

    except Exception as e:
        logging.error(f"Error writing to dataset: {e}")
        raise

    finally:
        logging.info(f"Closing tar file {tar_file}")
        tar_writer.close()
        for buffer in buffers:
            buffer.close()

    end_time = time.time()
    logging.info("Time taken to write to dataset: " + str(end_time - start_time))
    return
