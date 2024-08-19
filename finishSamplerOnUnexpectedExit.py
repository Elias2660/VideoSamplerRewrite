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
from WriteToDataset import write_to_dataset


def main():
    file_list = []

    try:
        start = time.time()
        parser = argparse.ArgumentParser(
            description="Create tar files even if the script is interrupted"
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
            "--out-channels",
            type=int,
            default=1,
            help="The number of output channels, default=1",
        )
        parser.add_argument(
            "--debug", type=bool, default=False, help="Debug mode, default false"
        )
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug mode activated")
        
        # find all dataset_*.csv files
        command = f"ls {os.path.join(args.dataset_path, args.dataset_search_string)}"
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        file_list = sorted(
            [ansi_escape.sub("", line).strip() for line in result.stdout.splitlines()]
        )

        logging.info(f"File List: {file_list}")
        
        # combines the dataframes
        total_dataframe = pd.DataFrame()
        for file in file_list:
            df = pd.read_csv(file)
            df["data_file"] = file
            total_dataframe = pd.concat([total_dataframe, df])

        # Batch directory operations
        for file in file_list:
            base_name = file.replace(".csv", "")

        data_frame_list = [group for _, group in total_dataframe.groupby("file")]
        for dataset in data_frame_list:
            dataset.reset_index(drop=True, inplace=True)

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
