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
"""

import time
import pandas as pd
import logging
from SamplerFunctions import sample_video
from WriteToDataset import write_to_dataset
import argparse
import subprocess
import multiprocessing
from multiprocessing import freeze_support
import concurrent
import re
import os


def main():
    file_list = []
    try:
        prep_file = open("dataprep.log", "r+")
        prep_file.truncate(0)
        prep_file.close()
    except:
        logging.info("prep file not found")
    try:
        start = time.time()
        parser = argparse.ArgumentParser(
            description="Prepare datasets for Deep Neural Network (DNN) training using video data."
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            help="Path to the dataset, defaults to .",
            default=".",
        )
        parser.add_argument(
            "--dataset-search-string",
            type=str,
            help="Grep string to get the datasets, defaults to dataset_*.csv",
            default="dataset_*.csv",
        )
        parser.add_argument(
            "--number-of-samples",
            type=int,
            help="the number of samples max that will be gathered by the sampler, defalt=1000",
            default=1000,
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            help="The number of workers to use for the multiprocessing, default=15",
            default=15,
        )
        parser.add_argument(
            "--frames-per-sample",
            type=int,
            help="The number of frames per sample, default=1",
            default=1,
        )
        parser.add_argument(
            "--normalize",
            type=bool,
            help="Normalize the images, default=True",
            default=True,
        )
        parser.add_argument(
            "--out-channels",
            type=int,
            help="The number of output channels, default=1",
            default=1,
        )
        parser.add_argument(
            "--debug",
            type=bool,
            help="Debug mode, default false",
            default=False,
        )
        parser.add_argument(
            "--crop",
            help="Crop the image, default=False",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--x-offset",
            type=int,
            help="The x offset for the crop, default=0",
            default=0,
        )
        parser.add_argument(
            "--y-offset",
            type=int,
            help="The y offset for the crop, default=0",
            default=0,
        )
        parser.add_argument(
            "--out-width",
            type=int,
            help="The width of the output image, default=400",
            default=400,
        )
        parser.add_argument(
            "--out-height",
            type=int,
            help="The height of the output image, default=400",
            default=400,
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
            subprocess.run(
                f"rm -rf {file.replace('.csv', '')}_samplestemporary", shell=True
            )
            subprocess.run(
                f"mkdir {file.replace('.csv', '')}_samplestemporary", shell=True
            )
            subprocess.run(
                f"rm -rf {file.replace('.csv', '')}_samplestemporarytxt", shell=True
            )
            subprocess.run(
                f"mkdir {file.replace('.csv', '')}_samplestemporarytxt", shell=True
            )

        data_frame_list = [group for _, group in total_dataframe.groupby("file")]
        for dataset in data_frame_list:
            dataset.reset_index(drop=True, inplace=True)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.max_workers, multiprocessing.cpu_count())
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
        try:
            result = subprocess.run(
                "ls *temporary", shell=True, capture_output=True, text=True
            )
            text = ansi_escape.sub("", result.stdout).split()
            logging.info(f"Samples sampled: {text}")
        except Exception as e:
            logging.error(f"An error occured in subprocess: {e}")
            raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.max_workers, multiprocessing.cpu_count())
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
            logging.info(f"Submitted {len(futures)} tasks to the executor")

        end = time.time()
        logging.info(f"Time taken to run the the script: {end - start} seconds")

    except Exception as e:
        logging.error(f"An error occured in main function: {e}")
        raise e

    finally:

        for file in file_list:
            subprocess.run(
                f"rm -rf {file.replace('.csv', '')}_samplestemporary", shell=True
            )
            subprocess.run(
                f"rm -rf {file.replace('.csv', '')}_samplestemporarytxt", shell=True
            )


if __name__ == "__main__":
    freeze_support()
    main()
