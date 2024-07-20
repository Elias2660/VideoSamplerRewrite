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
from multiprocessing import Manager, Semaphore, freeze_support, Lock
import concurrent  # for multitprocessing and other stuff
import re
import cv2
import os
from time import sleep
max_open_files = 100

file_semaphore = Semaphore(max_open_files)

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
    """
    get all the samples from sample video and writem them to a tar file using webdataset

    - read in the dataset
    - run the command off the dataset
    - write the samples to a tar file
    """
    try:
        logging.info(os.path.join(dataset_path, dataset_name.replace(".csv", ".tar")))
        with Manager() as manager:
            with file_semaphore:
                sample_list = manager.list()
                tar_lock = manager.Lock()
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(max_workers, int(multiprocessing.cpu_count() / 5)), initializer=cv2.setNumThreads, initargs=(1,)
                ) as executor_inner:
                    futures = [
                        executor_inner.submit(
                            sample_video,
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
                        )
                        for index, row in dataset.iterrows()
                    ]
                    sleep(3)
                    concurrent.futures.wait(futures)
                    sleep(3)
                    logging.info(
                        f"Submitted {len(futures)} tasks to the executor for {dataset_name}"
                    )
                    logging.info(f"Executor mapped for {dataset_name}")

            logging.info(
                f"Submitted {len(futures)} tasks to the executor for {dataset_name}"
            )
            logging.info(f"Executor mapped for {dataset_name}")

            logging.info(f"Writing samples to the tar file for {dataset_name}")
            logging.debug(f"Sampler list: {sample_list}")
            sleep(3)
            write_to_dataset(
                dataset_name.replace(".csv", ".tar"),
                sample_list,
                frames_per_sample,
                out_channels,
            )
            sleep(3)
        sample_end = time.time()
        logging.info(
            f"Time taken to write the samples for {dataset_name}: {sample_end - sample_start} seconds"
        )
        return futures
    except Exception as e:
        logging.error(f"An error occured in create_writers function: {e}")
        raise e


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

        create_writers(
            dataset_path,
            file_list[0],
            pd.read_csv(file_list[0]),
            number_of_samples,
            args.max_workers,
            args.frames_per_sample,
            args.normalize,
            args.out_channels,
        )
        # with Manager() as manager:
        #     with concurrent.futures.ProcessPoolExecutor(
        #         max_workers=args.max_workers
        #     ) as executor:
        #         logging.debug(f"Executor established")
        #         futures = [
        #             executor.submit(
        #                 create_writers,
        #                 dataset_path,
        #                 file,
        #                 pd.read_csv(file),
        #                 number_of_samples,
        #                 args.max_workers,
        #                 args.frames_per_sample,
        #                 args.normalize,
        #                 args.out_channels,
        #             )
        #             for file in file_list
        #         ]
        #         concurrent.futures.wait(futures)
        #         logging.debug(f"Executor mapped")
        end = time.time()
        logging.info(f"Time taken to run the the script: {end - start} seconds")

    except Exception as e:
        logging.error(f"An error occured in main function: {e}")
        raise e


if __name__ == "__main__":
    freeze_support()
    cv2.setNumThreads(1)
    """
    Run three 
    """
    main()
