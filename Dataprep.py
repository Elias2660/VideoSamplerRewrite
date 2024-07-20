import numpy as np
import time
import pandas as pd
import logging
import webdataset as wds
from SamplerFunctions import sample_video
from WriteToDataset import write_to_dataset
import argparse
import subprocess
import multiprocessing
from multiprocessing import Manager, freeze_support, Lock
import concurrent  # for multitprocessing and other stuff
import re
import cv2
import os


import os

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

logging.info(f"RLIMIT_NOFILE: {resource.getrlimit(resource.RLIMIT_NOFILE)}")


multiprocessing.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "1"


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def create_writers(
    dataset_path: str,
    dataset_name: str,
    dataset: pd.DataFrame,
    number_of_samples_max: int,
    max_workers: int,
    frames_per_sample: int,
    normalize: bool,
    out_channels: int,
):
    sample_start = time.time()
    try:
        logging.info(os.path.join(dataset_path, dataset_name.replace(".csv", ".tar")))
        with Manager() as manager:
            sample_list = manager.list()
            tar_lock = Manager().Lock()
            tar_lock = Manager().Lock()
            with multiprocessing.Pool(processes=max_workers) as pool:
                futures = [
                    pool.apply_async(
                        sample_video,
                        (
                            row["file"],
                            sample_list,
                            number_of_samples_max,
                            dataset_name.replace(".csv", ".tar"),
                            tar_lock,
                            row,
                            frames_per_sample,
                            frames_per_sample,
                            normalize,
                            out_channels,
                        ),
                    )
                    for index, row in dataset.iterrows()
                ]
                pool.close()
                pool.join()
                logging.info(
                    f"Submitted {len(futures)} tasks to the pool for {dataset_name}"
                )
                logging.info(f"Executor mapped for {dataset_name}")

            logging.info(f"Writing samples to the tar file for {dataset_name}")
            logging.debug(f"Sampler list: {sample_list}")
            write_to_dataset(
                dataset_name.replace(".csv", ".tar"),
                sample_list,
                frames_per_sample,
                out_channels,
            )
        sample_end = time.time()
        logging.info(
            f"Time taken to write the samples for {dataset_name}: {sample_end - sample_start} seconds"
        )
    except Exception as e:
        logging.error(f"An error occured in create_writers function: {e}")
        raise e
    return futures


def main():
    try:

        start = time.time()
        parser = argparse.ArgumentParser(
            description="Perform data preparation for DNN training on a video set."
        )

        parser.add_argument(
            "--dataset_path", type=str, help="Path to the datasets", default="."
        )
        parser.add_argument(
            "--dataset-search-string",
            type=str,
            help="Grep string to get the datasets",
            default="dataset_*.csv",
        )
        parser.add_argument(
            "--number-of-samples",
            type=int,
            help="the number of samples max that will be gathered by the sampler",
            default=1000,
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            help="The number of workers to use for the multiprocessing",
            default=50,
        )
        parser.add_argument(
            "--frames-per-sample",
            type=int,
            help="The number of frames per sample",
            default=1,
        )
        parser.add_argument(
            "--normalize", type=bool, help="Normalize the images", default=True
        )
        parser.add_argument(
            "--out-channels", type=int, help="The number of output channels", default=1
        )
        args = parser.parse_args()
        dataset_path = args.dataset_path
        number_of_samples = args.number_of_samples
        command = f"ls {os.path.join(args.dataset_path, args.dataset_search_string)}"
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        file_list = sorted(
            [ansi_escape.sub("", line).strip() for line in result.stdout.splitlines()]
        )
        logging.info(f"File List: {file_list}")
        with Manager() as manager:
            pool =  multiprocessing.Pool(processes=args.max_workers)
            logging.debug(f"Pool established")
            results = [
                pool.apply_async(
                    create_writers,
                    (
                        dataset_path,
                        file,
                        pd.read_csv(file),
                        number_of_samples,
                        args.max_workers,
                        args.frames_per_sample,
                        args.normalize,
                        args.out_channels,
                    ),
                )
                for file in file_list
            ]
            for result in results:
                result.get()
            logging.debug(f"Pool mapped")
            end = time.time()
            logging.info(f"Time taken to run the script: {end - start} seconds")
    except Exception as e:
        logging.error(f"An error occurred in main function: {e}")
        raise e


if __name__ == "__main__":
    # cv2.setNumThreads(400)
    freeze_support()
    """
    Run three 
    """
    main()
