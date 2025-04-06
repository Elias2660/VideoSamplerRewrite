"""
WriteToDataset.py

This module provides functionality to write samples from a directory to a dataset tar file.

Functions:
    write_to_dataset(directory: str, tar_file: str, frames_per_sample: int = 1, out_channels: int = 1, batch_size: int = 10) -> None:
        Writes samples from a directory to a dataset tar file.

Dependencies:
    - webdataset
    - os
    - logging
    - time
    - torch
    - torchvision
    - io
    - concurrent.futures

Raises:
    Exception: If there is an error writing to the dataset.

License:
    This project is licensed under the MIT License - see the LICENSE file for details.
"""
import datetime
import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import webdataset as wds
from torchvision import transforms


def process_sample(file, directory, frames_per_sample, out_channels):
    """

    :param file: param directory:
    :param frames_per_sample: param out_channels:
    :param directory: param out_channels:
    :param out_channels:

    """
    # convert the sample into something that can be read into the tar files
    try:
        data = np.load(os.path.join(directory, file))
        np_tensor = data["tensor"]
        frame = torch.from_numpy(np_tensor)

        s_c_file_path = os.path.join(directory + "txt",
                                     file.replace(".npz", ".txt"))
        with open(s_c_file_path, "r") as s_c_file:
            s = file.replace(".npz", "").split("/")[-1].split("_")
            if len(s) != 4:
                logging.error(
                    f"Unexpected format in file name: {file}, split result: {s}"
                )
                return None
            filename, d_name, _, _ = s
            video_path = filename.replace("SPACE", " ")
            sample_class = d_name
            frame_num = s_c_file.read().split("-")

        # to save space, immediately delete the sample's .npz and .txt file
        os.remove(os.path.join(directory, file))
        os.remove(s_c_file_path)

        base_name = os.path.basename(video_path).replace(" ", "_").replace(
            ".", "_")
        video_time = os.path.basename(video_path).split(".")[0]
        time_sec = time.mktime(time.strptime(video_time, "%Y-%m-%d %H:%M:%S"))
        time_struct = time.localtime(time_sec + int(frame_num[0]) // 3)
        curtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
        metadata = f"{video_path},{frame_num[0]},{curtime}"

        sample = {
            "__key__": "_".join((base_name, "_".join(frame_num))),
            "cls": str(sample_class).encode("utf-8"),
            "metadata.txt": metadata.encode("utf-8"),
        }

        # write sample / image to memory
        buffers = []
        for i in range(frames_per_sample):
            img = transforms.ToPILImage()(
                frame[i] / 255.0).convert(  # tar files are written as pngs
                    "RGB" if out_channels == 3 else "L")
            buf = io.BytesIO()  # saving the images to memory
            img.save(fp=buf, format="png")
            buffers.append(buf.getbuffer())

        for i, buffer in enumerate(buffers):
            sample[f"{i}.png"] = buffer

        return sample
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e):
            # this is where the file is corrupted because the tensor wasn't read properly
            logging.error(
                f"PytorchStreamReader error processing sample {file}: {e}")
        else:
            logging.error(f"RuntimeError processing sample {file}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing sample {file}: {e}")
        return None


def write_to_dataset(
    directory: str,
    tar_file: str,
    dataset_path: str,
    frames_per_sample: int = 1,
    out_channels: int = 1,
    batch_size: int = 60,
    equalize_samples: bool = False,
    max_workers_tar_writing: int = 4,
):
    """Writes samples from a directory to a dataset tar file.

    :param directory: str
    :param tar_file: str
    :param frames_per_sample: int
    :param out_channels: int
    :param batch_size: int
    :param num_workers: int
    :param Raises: param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)
    :param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)
    :param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)
    :param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)

    """
    try:
        tar_writer = wds.TarWriter(tar_file, encoder=False)
        start_time = time.time()

        file_list = [
            f for f in os.listdir(directory) if not f.endswith(".txt")
        ]

        # equalization to ensure the number of samples per class in each sample is
        # equal to each other (BUT NOT EQUALIZING SAMPLES ACROSS TAR FILES, THOSE
        # ARE INDEPENDENT)
        if equalize_samples:
            logging.info(f"Equalizing samples for {directory}")
            sample_dict = {}
            # first find the class with the least number of samples
            # then for each class, delete samples until the number
            # of samples is equal to the minimum
            for file in file_list:
                s = file.replace(".npz", "").split("/")[-1].split("_")
                _, sample_class, _, _ = s
                if sample_class in sample_dict:
                    sample_dict[sample_class].append(file)
                else:
                    sample_dict[sample_class] = [file]
            min_samples = min(
                [len(samples) for samples in sample_dict.values()])
            logging.info(
                f"Minimum number of samples for directory {directory}: {min_samples}"
            )
            for samples in sample_dict.values():
                random.shuffle(samples)
                for sample in samples[min_samples:]:
                    os.remove(os.path.join(directory, sample))
                    os.remove(
                        os.path.join(directory + "txt",
                                     sample.replace(".npz", ".txt")))
            logging.info(
                f"Equalized samples for {directory} and {directory + 'txt'}")

        file_list = [
            f for f in os.listdir(directory) if not f.endswith(".txt")
        ]
        file_size = len(file_list)
        logging.info(
            f"Reading in the samples from {directory}, finding {len(file_list)} files"
        )

        sample_count = 0  # for logging purposes
        old_time = time.time()
        # using threadpool because ilab is stingy with multiple processes
        # yes, I know about GIL lock
        with ThreadPoolExecutor(
                max_workers=max_workers_tar_writing) as executor:
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i:i + batch_size]
                results = list(
                    executor.map(
                        # use batching here too, to speed up the process
                        process_sample,
                        batch,
                        [directory] * len(batch),
                        [frames_per_sample] * len(batch),
                        [out_channels] * len(batch),
                    ))
                for sample in results:
                    if sample:
                        tar_writer.write(sample)
                        sample_count += 1
                        if sample_count % 30000 == 0 and sample_count != 0:
                            new_time = time.time()
                            logging.info(
                                f"Writing sample {sample_count} to dataset tar file {tar_file}; time to write 30,000 samples: {((new_time - old_time)/30000):.2f} second(s) per sample"
                            )
                            old_time = new_time
        # make sure everything is written
        executor.shutdown(wait=True)
    except Exception as e:
        executor.shutdown(wait=False)
        logging.error(f"Error writing to dataset: {e}")
        raise e

    finally:
        # tar writer MUST CLOSE, or the data is unusable
        logging.info(f"Closing tar file {tar_file}")
        tar_writer.close()

    # logging into the RUN_DESCRIPTION
    with open(os.path.join(dataset_path, "RUN_DESCRIPTION.log"), "a+") as rd:
        rd.write(f"{file_size} samples collected by tar file {tar_file}\n")

    end_time = time.time()
    logging.info(
        f"Time taken to write to {tar_file}: {str(datetime.timedelta(seconds=int(end_time - start_time)))}"
    )
    logging.info(f"The number of samples in {tar_file}: {file_size}")
    return
