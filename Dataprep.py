"""
Module Name: Dataprep.py

Description:
    Samples frames from video-derived dataset CSVs for DNN training. 
    - Searches for dataset files matching a glob pattern.
    - Aggregates all matched CSVs into a single DataFrame grouped by video file.
    - Creates temporary directories for each dataset.
    - Uses parallel sampling (SamplerFunctions.sample_video) to extract frames.
    - Packages sampled images into tar archives (WriteToDataset.write_to_dataset).
    - Logs progress, execution time, and manages file permissions.
    - Cleans up all temporary folders on completion or error.

Usage:
    python Dataprep.py \
      --video-input-path <video_dir> \
      --dataset-input-path <csv_dir> \
      --out-path <output_dir> \
      --dataset-search-string "<pattern>" \
      --number-of-samples <max_samples> \
      --frames-per-sample <frames_per_sample> \
      [--max-workers <workers>] \
      [--normalize <True|False>] \
      [--out-channels <channels>] \
      [--crop] \
      [--x-offset <px>] \
      [--y-offset <px>] \
      [--out-width <px>] \
      [--out-height <px>] \
      [--equalize-samples] \
      [--dataset-writing-batch-size <size>] \
      [--max-threads-pic-saving <threads>] \
      [--max-workers-tar-writing <workers>] \
      [--max-batch-size-sampling <size>] \
      [--debug]

Arguments:
    --video-input-path           Path to source `.mp4` videos (default: `.`).
    --dataset-input-path         Path to directory of dataset CSVs (default: `.`).
    --out-path                   Destination for tar files and logs (default: `.`).
    --dataset-search-string      Glob for dataset CSV filenames (default: `dataset_*.csv`).
    --number-of-samples          Max samples per dataset (default: 40000).
    --frames-per-sample          Frames per sample (default: 1).
    --max-workers                Parallel sampling processes (default: 15).
    --normalize                  Normalize images (default: True).
    --out-channels               Output image channels (default: 1).
    --crop                       Enable cropping (default: False).
    --x-offset, --y-offset       Crop offsets in pixels (default: 0).
    --out-width, --out-height    Crop dimensions (required if `--crop`).
    --equalize-samples           Balance class sample counts (default: False).
    --dataset-writing-batch-size Write batch size for archives (default: 20).
    --max-threads-pic-saving     Threads for saving images (default: 20).
    --max-workers-tar-writing    Parallel tar-writing processes (default: 4).
    --max-batch-size-sampling    Sampling batch size per task (default: 20).
    --debug                      Enable debug-level logging (default: False).

Dependencies:
    - SamplerFunctions.sample_video
    - WriteToDataset.write_to_dataset
    - pandas, argparse, concurrent.futures, multiprocessing, logging, subprocess, os, re, time, datetime

Behavior:
    1. Locate and read all CSVs matching `--dataset-search-string`.
    2. Combine into one DataFrame, grouped by video file.
    3. Create per-dataset temporary directories.
    4. Sample frames in parallel and write to temp folders.
    5. Record sampling results in RUN_DESCRIPTION.log.
    6. Package each temp folder into a `.tar` archive in `--out-path`.
    7. Set file permissions on archives.
    8. Remove temporary directories before exiting.
"""


import argparse
import concurrent.futures
import datetime
import logging

logging.basicConfig(
    format="%(asctime)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
import os
import re
import subprocess
import time
from multiprocessing import freeze_support

import pandas as pd

from SamplerFunctions import sample_video
from WriteToDataset import write_to_dataset


def main():
    try:
        start = time.time()
        parser = argparse.ArgumentParser(
            description="Prepare datasets for Deep Neural Network (DNN) training using video data."
        )
        parser.add_argument(
            "--video-input-path",
            type=str,
            default=".",
            help="Path for the input video files (the .mp4 files), defaults to .",
        )
        parser.add_argument(
            "--dataset-input-path",
            type=str,
            default=".",
            help="path for the input dataset files (dataset_*.csv)"
        )
        parser.add_argument(
            "--out-path",
            type=str,
            default=".",
            help="path for the output files"
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
            "--debug", action="store_true", help="Debug mode, default false"
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
            default=None,
            help="The width of the output image, default=None NOTE: if you set crop to true you cannot set these to none",
        )
        parser.add_argument(
            "--out-height",
            type=int,
            default=None,
            help="The height of the output image, default=None NOTE: if you set crop to true you cannot set these to none",
        )
        parser.add_argument(
            "--scale-factor",
            type=float,
            default=1.0,
            help="The scaling factor to scale resultant sample images"
        )
        parser.add_argument(
            "--equalize-samples",
            action="store_true",
            default=False,
            help="Equalize the samples so that each class has the same number of samples, default=False",
        )
        parser.add_argument(
            "--dataset-writing-batch-size",
            type=int,
            default=20,
            help="The batch size for writing to the dataset, default=20",
        )
        parser.add_argument(
            "--max-threads-pic-saving",
            type=int,
            default=20,
            help="The maximum number of threads to use for saving the pictures, default=20",
        )
        parser.add_argument(
            "--max-workers-tar-writing",
            type=int,
            default=4,
            help="The maximum number of workers to use for writing to the tar file, default=4",
        )
        parser.add_argument(
            "--max-batch-size-sampling",
            type=int,
            default=20,
            help="The maximum batch size for sampling the video, default=20",
        )

        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug mode activated")

        # logMAXXing
        logging.info(
            f"Starting the data preparation process, with frames per sample: {args.frames_per_sample}, number of samples: {args.number_of_samples}, and max workers: {args.max_workers}, equalize samples: {args.equalize_samples}"
        )
        logging.info(f"Video input path: {args.video_input_path}")
        logging.info(f"Dataset input path {args.dataset_input_path}")
        logging.info(f"Output path: {args.out_path}")
        logging.info(f"Normalize: {args.normalize}")
        logging.info(f"Output channels: {args.out_channels}")
        logging.info(f"Debug mode: {args.debug}")
        logging.info(f"X offset: {args.x_offset}")
        logging.info(f"Y offset: {args.y_offset}")
        logging.info(f"Output width: {args.out_width}")
        logging.info(f"Output height: {args.out_height}")
        logging.info(f"Equalize samples: {args.equalize_samples}")
        logging.info(f"Dataset writing batch size: {args.dataset_writing_batch_size}")
        logging.info(f"Max threads for picture saving: {args.max_threads_pic_saving}")
        logging.info(f"Max workers for tar writing: {args.max_workers_tar_writing}")
        logging.info(f"Max batch size for sampling: {args.max_batch_size_sampling}")
        logging.info(f"Crop has been set as {args.crop}")

        number_of_samples = args.number_of_samples
        # find all dataset_*.csv files
        file_list = [file for file in os.listdir(args.dataset_input_path) if bool(re.search(r'\d', file)) and file.startswith("dataset_") and file.endswith(".csv")]
        logging.info(f"File List: {file_list}")
        
        if len(file_list) == 0:
            raise Exception("There are no dataset_*.csv files found. Try to specify the right path or actually create the files.`")
        
        # combines the dataframes
        total_dataframe = pd.DataFrame()
        for file in file_list:
            df = pd.read_csv(os.path.join(args.dataset_input_path, file))
            df["data_file"] = file
            total_dataframe = pd.concat([total_dataframe, df])

        # Batch directory operations
        for file in file_list:
            base_name = file.replace(".csv", "")
            os.makedirs(os.path.join(args.out_path, f"{base_name}_samplestemporary"), exist_ok=True)
            os.makedirs(os.path.join(args.out_path, f"{base_name}_samplestemporarytxt"), exist_ok=True)

        data_frame_list = [group for _, group in total_dataframe.groupby("file")]
        for dataset in data_frame_list:
            dataset.reset_index(drop=True, inplace=True)

        # change the permissions for the directories so that everybody can determine progress for the files
        subprocess.run(f"chmod 777 {os.path.join(args.out_path, '*temporary*')}", shell=True)
        subprocess.run(f"chmod 777 {os.path.join(args.out_path, 'dataprep.log')}", shell=True)

        try:
            # for each dataset which has the samples to gather from the video, sample the video
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.max_workers, os.cpu_count())
            ) as executor:
                futures = [
                    executor.submit(
                        sample_video,
                        args.video_input_path,
                        args.dataset_input_path,
                        args.out_path,
                        dataset.loc[0, "file"],
                        dataset,
                        number_of_samples,
                        args.frames_per_sample,
                        args.normalize,
                        args.out_channels,
                        args.frames_per_sample,
                        args.out_height,
                        args.out_width,
                        args.x_offset,
                        args.y_offset,
                        args.crop,
                        args.scale_factor,
                        args.max_batch_size_sampling,
                        args.max_threads_pic_saving,
                    )
                    for dataset in data_frame_list
                ]
                concurrent.futures.wait(futures)
                logging.info(f"Submitted {len(futures)} tasks to the executor")
                executor.shutdown(
                    wait=True
                )  # make sure all the sampling finishes; don't want half written samples
        except Exception as e:
            logging.error(f"An error occurred in the executor: {e}")
            executor.shutdown(wait=False)
            raise e

        # log header which will be filled out by the write_to_dataset functions
        with open(os.path.join(args.# The code seems to be defining a variable `dataset_input_path` in
        # Python, but it is not assigned any value.
        dataset_input_path, "RUN_DESCRIPTION.log"), "a+") as rd:
            rd.write("\n-- Sample Collection Results --\n")

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.max_workers, os.cpu_count())
            ) as executor:
                futures = [
                    executor.submit(
                        write_to_dataset,
                        file.replace(".csv", "") + "_samplestemporary",
                        file.replace(".csv", ".tar"),
                        args.dataset_input_path,
                        args.out_path,
                        args.frames_per_sample,
                        args.out_channels,
                        args.dataset_writing_batch_size,
                        args.equalize_samples,
                        args.max_workers_tar_writing,
                    )
                    for file in file_list
                ]
                concurrent.futures.wait(futures)

            end = time.time()
            logging.info(
                f"Time taken to run the script: {datetime.timedelta(seconds=int(end - start))} seconds"
            )
            # make sure all of the writing is done
            executor.shutdown(wait=True)
            subprocess.run(f"chmod -R 777 {os.path.join(args.out_path, '*.tar')}", shell=True)
        except Exception as e:
            logging.error(f"An error occurred in the executor: {e}")
            executor.shutdown(wait=False)
            raise e

    finally:
        # deconstruct all resources and declutter data
        for file in file_list:
            base_name = file.replace(".csv", "")
            os.rmdir(os.path.join(args.out_path, f'{base_name}_samplestemporary'))
            os.rmdir(os.path.join(args.out_path, f"{base_name}_samplestemporarytxt"))


if __name__ == "__main__":
    freeze_support()  # needed for windows
    main()
