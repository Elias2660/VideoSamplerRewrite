"""
Dataprep.py

This script prepares datasets for Deep Neural Network (DNN) training using video data. It performs the following tasks:
1. Clears the existing log file or creates a new one if it doesn't exist.
2. Parses command-line arguments to configure the data preparation process.
3. Uses a thread pool to concurrently process video files and write the processed data to a dataset.
4. Logs the progress and execution time of the data preparation process.
5. Cleans up temporary files created during the process.

Functions:
- main(): The main function that orchestrates the data preparation process.

Usage:
    python Dataprep.py --dataset_path <path-to-dataset> --dataset_name <dataset-name> --number_of_samples_max <max-samples> --max_workers <number-of-workers> --frames_per_sample <frames-per-sample>

Dependencies:
- pandas
- argparse
- subprocess
- multiprocessing
- concurrent.futures
- re
- os
- logging
- SamplerFunctions.sample_video
- WriteToDataset.write_to_dataset

Example:
    python Dataprep.py --dataset_path ./data --dataset_name my_dataset --number_of_samples_max 1000 --max_workers 4 --frames_per_sample 10

Raises:
    Exception: If there is an error in the data preparation process.

License:
    This project is licensed under the MIT License - see the LICENSE file for details.
"""


import os
import re
import time
import logging
import argparse
import pandas as pd
import datetime
import subprocess
import concurrent.futures
from multiprocessing import freeze_support
from SamplerFunctions import sample_video
from WriteToDataset import write_to_dataset


def main():
    file_list = []
    try:
        with open("dataprep.log", "w") as prep_file:
            prep_file.truncate(0)
    except FileNotFoundError:
        logging.info("prep file not found")

    try:
        start = time.time()
        parser = argparse.ArgumentParser(
            description="Prepare datasets for Deep Neural Network (DNN) training using video data."
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            default=".",
            help="Path to the dataset, defaults to .",
        )
        parser.add_argument(
            "--dataset-search-string",
            type=str,
            default="dataset_*.csv",
            help="Grep string to get the datasets, defaults to dataset_*.csv",
        )
        parser.add_argument(
            "--number-of-samples",
            type=int,
            default=40000,
            help="the number of samples max that will be gathered by the sampler, default=1000",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=15,
            help="The number of workers to use for the multiprocessing, default=15",
        )
        parser.add_argument(
            "--frames-per-sample",
            type=int,
            default=1,
            help="The number of frames per sample, default=1",
        )
        parser.add_argument(
            "--normalize",
            type=bool,
            default=True,
            help="Normalize the images, default=True",
        )
        parser.add_argument(
            "--out-channels",
            type=int,
            default=1,
            help="The number of output channels, default=1",
        )
        parser.add_argument(
            "--debug", type=bool, default=False, help="Debug mode, default false"
        )
        parser.add_argument(
            "--crop",
            action="store_true",
            default=False,
            help="Crop the image, default=False",
        )
        parser.add_argument(
            "--x-offset",
            type=int,
            default=0,
            help="The x offset for the crop, default=0",
        )
        parser.add_argument(
            "--y-offset",
            type=int,
            default=0,
            help="The y offset for the crop, default=0",
        )
        parser.add_argument(
            "--out-width",
            type=int,
            default=400,
            help="The width of the output image, default=400",
        )
        parser.add_argument(
            "--out-height",
            type=int,
            default=400,
            help="The height of the output image, default=400",
        )

        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug mode activated")

        logging.info(
            f"Starting the data preparation process, with frames per sample: {args.frames_per_sample}, number of samples: {args.number_of_samples}, and max workers: {args.max_workers}"
        )
        logging.info(f"Crop has been set as {args.crop}")

        number_of_samples = args.number_of_samples
        command = f"ls {os.path.join(args.dataset_path, args.dataset_search_string)}"
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        file_list = sorted(
            [ansi_escape.sub("", line).strip() for line in result.stdout.splitlines()]
        )

        logging.info(f"File List: {file_list}")

        total_dataframe = pd.DataFrame()
        for file in file_list:
            df = pd.read_csv(file)
            df["data_file"] = file
            total_dataframe = pd.concat([total_dataframe, df])

        # Batch directory operations
        for file in file_list:
            base_name = file.replace(".csv", "")
            os.makedirs(f"{base_name}_samplestemporary", exist_ok=True)
            os.makedirs(f"{base_name}_samplestemporarytxt", exist_ok=True)

        data_frame_list = [group for _, group in total_dataframe.groupby("file")]
        for dataset in data_frame_list:
            dataset.reset_index(drop=True, inplace=True)

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.max_workers, os.cpu_count())
            ) as executor:
                futures = [
                    executor.submit(
                        sample_video,
                        dataset.loc[0, "file"],
                        dataset,
                        number_of_samples,
                        args.frames_per_sample,
                        args.normalize,
                        args.out_channels,
                        args.frames_per_sample,
                    )
                    for dataset in data_frame_list
                ]
                concurrent.futures.wait(futures)
                logging.info(f"Submitted {len(futures)} tasks to the executor")
            executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f"An error occurred in the executor: {e}")
            executor.shutdown(wait=False)
            raise e

        try:
            result = subprocess.run(
                "ls *temporary", shell=True, capture_output=True, text=True
            )
            text = ansi_escape.sub("", result.stdout).split()
            logging.debug(f"Samples sampled: {text}")
        except Exception as e:
            logging.error(f"An error occurred in subprocess: {e}")
            raise e

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.max_workers, os.cpu_count())
            ) as executor:
                futures = [
                    executor.submit(
                        write_to_dataset,
                        file.replace(".csv", "") + "_samplestemporary",
                        file.replace(".csv", ".tar"),
                        args.frames_per_sample,
                        args.out_channels,
                    )
                    for file in file_list
                ]
                concurrent.futures.wait(futures)

            end = time.time()
            logging.info(
                f"Time taken to run the script: {datetime.timedelta(seconds=int(end - start))} seconds"
            )
            executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f"An error occurred in the executor: {e}")
            executor.shutdown(wait=False)
            raise e

    except Exception as e:
        logging.error(f"An error occurred in main function: {e}")
        raise e

    finally:
        for file in file_list:
            base_name = file.replace(".csv", "")
            os.rmdir(f"{base_name}_samplestemporary")
            os.rmdir(f"{base_name}_samplestemporarytxt")


if __name__ == "__main__":
    freeze_support()
    main()
