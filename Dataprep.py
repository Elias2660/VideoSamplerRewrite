import math
import numpy as np
import torch
import time
import pandas as pd
import logging
import webdataset as wds
from SamplerFunctions import sample_video
import argparse
import subprocess
from multiprocessing import Manager, freeze_support, Lock
import concurrent # for multitprocessing and other stuff

from SamplerFunctions import sample_video

import re
import os


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def create_writers(dataset_path, dataset_name, dataset):
    sample_start = time.time()
    """
    get all the samples from sample video and writem them to a tar file using webdataset

    - read in the dataset
    - run the command off the dataset
    - write the samples to a tar file
    """
    datawriter = wds.TarWriter(os.path.join(dataset_path, dataset_name.replace(".csv", ".tar")), encoder=False)
    write_list = Manager().list()
    tar_lock = Lock()
    
    
    # run the sampling call here!!!
    
    
    sample_end = time.time()
    
    
    logging.info(f"Time taken to write the samples for {dataset_name}: {sample_end - sample_start} seconds")
    

if __name__ == "__main__":
    freeze_support()
    """
    Run three 
    """
    try:
            
        start = time.time()
        parser = argparse.ArgumentParser(
        description="Perform data preparation for DNN training on a video set.")
        
        parser.add_argument("--dataset_path", type=str, help="Path to the datasets", default=".")
        parser.add_arguement("--dataset-search-string", type=str, help="Grep string to get the datasets", default = "dataset_*.csv")
        
        args = parser.parse_args()
        
        
        command = f"ls {args.dataset_path} | grep {args.dataset_search_string}"
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        file_list = sorted(
            [ansi_escape.sub("", line).strip() for line in result.stdout.splitlines()]
        )
        logging.info(f"File List: {file_list}")
        
        
        
        
        end = time.time()
        logging.info(f"Time taken to run the the script: {end - start} seconds")
        
    except Exception as e:
        logging.error(f"An error occured: {e}")
        raise e
