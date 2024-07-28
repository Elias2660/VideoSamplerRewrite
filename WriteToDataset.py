import webdataset as wds
import os
import logging
import time
import torch
from torchvision import transforms
import io
import time


def write_to_dataset(
    directory: str,
    tar_file: str,
    frames_per_sample: int = 1,
    out_channels: int = 1,
):
    """
    Writes samples from a directory to a dataset tar file.

    Args:
        directory (str): The directory containing the samples.
        tar_file (str): The path to the output tar file.
        frames_per_sample (int, optional): The number of frames per sample. Defaults to 1.
        out_channels (int, optional): The number of output channels. Defaults to 1.

    Raises:
        Exception: If there is an error writing to the dataset.

    Returns:
        None
    """

    try:
        tar_writer = wds.TarWriter(tar_file, encoder=False)
        start_time = time.time()

        file_list = os.listdir(directory)
        logging.info(
            f"Reading in the samples from {directory}, finding {len(file_list)} files"
        )

        sample_count = 0  # for logging purposes
        for file in file_list:
            if file.endswith(".txt"):
                continue
            frame = torch.load(os.path.join(directory, file))
            s_c_file = open(
                os.path.join(directory + "txt", file.replace(".pt", ".txt")), "r"
            )
            s = file.replace(".pt", "").split("/")[-1].split("_")
            if len(s) != 4:
                logging.error(
                    f"Unexpected format in file name: {file}, split result: {s}"
                )
            filename, d_name, _, _ = s
            video_path = filename.replace("SPACE", " ")
            sample_class = d_name
            frame_num = s_c_file.read().split("-")
            s_c_file.close()

            # remove the s_c file and pt file
            os.remove(os.path.join(directory, file))
            os.remove(os.path.join(directory + "txt", file.replace(".pt", ".txt")))

            logging.info(
                f"video_path: {video_path}, sample_class: {sample_class}, frame_num: {frame_num}"
            )
            logging.debug(
                f"Writing sample to dataset; Frame shape: {frame.shape}, Frame number: {frame_num[0]}, Video path: {video_path}, frame type: {type(frame)}"
            )

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
                logging.info(f"Writing sample to dataset tar file")
                tar_writer.write(sample)
            else:
                buffers = []

                for i in range(frames_per_sample):
                    if 3 == out_channels:
                        img = transforms.ToPILImage()(frame[i] / 255.0).convert("RGB")
                    else:
                        img = transforms.ToPILImage()(frame[i] / 255.0).convert("L")

                    buffers.append(io.BytesIO())
                    img.save(fp=buffers[-1], format="png")

                sample = {
                    "__key__": "_".join((base_name, "_".join(frame_num))),
                    "cls": str(sample_class).encode("utf-8"),
                    "metadata.txt": metadata.encode("utf-8"),
                }
                for i in range(frames_per_sample):
                    sample[f"{i}.png"] = buffers[i].getbuffer()

                tar_writer.write(sample)

            if sample_count % 100 == 0:
                logging.info(f"Writing sample to dataset tar file")
            sample_count += 1

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
