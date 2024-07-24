"""
TODO: create dirs function -> standardize!
TODO: adapt writers function
TODO: adapt the sampler function

? MAYBE: Add multiprocessing???
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
import concurrent  # for multitprocessing and other stuff
import re
import os


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


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
        parser.add_argument("--bg-subtract", type=str,choices=["mog2", "knn"] ,help="The background subtraction method to use", default=None)
        
        args = parser.parse_args()
        number_of_samples = args.number_of_samples
        command = f"ls {os.path.join(args.dataset_path, args.dataset_search_string)}"
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        file_list = sorted(
            [ansi_escape.sub("", line).strip() for line in result.stdout.splitlines()]
        )
        
        
        logging.info(f"File List: {file_list}")
        counts = pd.read_csv("counts.csv")

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

        # group by file to get for each file a list of rows
        # then for each file, create a writer
        data_frame_list = [group for _, group in total_dataframe.groupby("file")]
        logging.info(len(data_frame_list))
        # The `data_frame_list` in the provided code is being used to store groups of rows from the
        # `total_dataframe` DataFrame.
        # logging.debug(data_frame_list)
        for dataset in data_frame_list:
            # reset dataframe index
            dataset.reset_index(drop=True, inplace=True)
        for i in range(3):
            logging.info(data_frame_list[i].head())
        with concurrent.futures.ThreadPoolExecutor(
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
                    args.bg_subtract,
                    args.frames_per_sample,
                )
                for dataset in data_frame_list
            ]
            concurrent.futures.wait(futures)
            logging.info(f"Submitted {len(futures)} tasks to the executor")
            logging.info(f"Executor mapped")

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
            logging.info(f"Executor mapped")

        end = time.time()
        logging.info(f"Time taken to run the the script: {end - start} seconds")

    except Exception as e:
        logging.error(f"An error occured in main function: {e}")
        raise e

    finally:
        # remove all the dirs
        for file in file_list:
            subprocess.run(
                f"rm -rf {file.replace('.csv', '')}_samplestemporary", shell=True
            )


if __name__ == "__main__":
    freeze_support()
    """
    Run three 
    """
    main()
