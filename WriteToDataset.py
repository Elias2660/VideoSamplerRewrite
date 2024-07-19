import webdataset as wds
import pandas as pd
import os
import logging
import time
import multiprocessing
from torchvision import transforms
import io


def write_to_dataset(
    tar_writer: wds.TarWriter,
    row: pd.Series,
    samples,
    lock: multiprocessing.Lock,
    video_path: str,
    frames_per_sample: int = 1,
    out_channels: int = 1,
):
    """using the lock, write the files to the datase5t"""
    start_time = time.time()
    logging.info(f"Enumerating through the sampes")
    for sample_num, sample in enumerate(samples):
        frame, video_path, frame_num = sample
        base_name = os.path.basename(video_path).replace(" ", "_").replace(".", "_")
        video_time = os.path.basename(video_path).split(".")[0]
        time_sec = time.mktime(time.strptime(video_time, "%Y-%m-%d %H:%M:%S"))
        time_struct = time.localtime(time_sec + int(frame_num[0]) // 3)
        curtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
        metadata = f"{video_path},{frame_num[0]},{curtime}"
        # height, width = frame.size(2), frame.size(3)
        if 1 == frames_per_sample:
            if 3 == out_channels:
                img = transforms.ToPILImage()(frame[0] / 255.0).convert("RGB")
            else:
                img = transforms.ToPILImage()(frame[0] / 255.0).convert("L")
                # Now save the image as a png into a buffer in memory
            buf = io.BytesIO()
            img.save(fp=buf, format="png")
            sample = {
                "__key__": "_".join((base_name, "_".join(frame_num))),
                "0.png": buf.getbuffer(),
                "cls": row["class"].encode("utf-8"),
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
                    # Now save the image as a png into a buffer in memory
                    buffers.append(io.BytesIO())
                    img.save(fp=buffers[-1], format="png")

                sample = {
                    "__key__": "_".join((base_name, "_".join(frame_num))),
                    "cls": row["class"].encode("utf-8"),
                    "metadata.txt": metadata.encode("utf-8"),
                }
                for i in range(frames_per_sample):
                    sample[f"{i}.png"] = buffers[i].getbuffer()
    with lock:
        tar_writer.write(sample)

    end_time = time.time()
    logging.info("Time taken to write to dataset: " + str(end_time - start_time))
