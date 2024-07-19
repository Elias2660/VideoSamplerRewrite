import numpy as np
import time
import pandas as pd
import logging
import webdataset as wds
from SamplerFunctions import sample_video
import argparse
import subprocess
from multiprocessing import freeze_support, Lock
import concurrent  # for multitprocessing and other stuff
from SamplerFunctions import sample_video
import re
import os


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def create_writers(
    dataset_path,
    dataset_name,
    dataset,
    number_of_samples_max,
    max_workers,
    frames_per_sample,
):
    sample_start = time.time()
    """
    get all the samples from sample video and writem them to a tar file using webdataset

    - read in the dataset
    - run the command off the dataset
    - write the samples to a tar file
    """
    try:

        datawriter = wds.TarWriter(
            os.path.join(dataset_path, dataset_name.replace(".csv", ".tar")),
            encoder=False,
        )
        # write_list = Manager().list()
        tar_lock = Lock()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    sample_video,
                    number_of_samples_max,
                    datawriter,
                    tar_lock,
                    row,
                    frames_per_sample,
                )
                for index, row in dataset.iterrows()
            ]
            concurrent.futures.wait(futures)
            logging.info(f"Executor mapped for {dataset_name}")

        sample_end = time.time()
        datawriter.close()
        logging.info(
            f"Time taken to write the samples for {dataset_name}: {sample_end - sample_start} seconds"
        )
    except Exception as e:
        logging.error(f"An error occured: {e}")
        raise e


if __name__ == "__main__":
    freeze_support()
    """
    Run three 
    """
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

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            logging.debug(f"Executor established")
            futures = [
                executor.submit(
                    create_writers,
                    dataset_path,
                    pd.read_csv(file),
                    number_of_samples,
                    args.max_workers,
                    args.frames_per_sample,
                )
                for file in file_list
            ]
            concurrent.futures.wait(futures)
            logging.debug(f"Executor mapped")
        end = time.time()
        logging.info(f"Time taken to run the the script: {end - start} seconds")

    except Exception as e:
        logging.error(f"An error occured: {e}")
        raise e
