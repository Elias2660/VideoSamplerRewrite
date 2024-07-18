import math
import numpy as np
import torch
import time
import pandas as pd
import logging
import os
import webdataset as wds
from SamplerFunctions import sample_video
import argparse
import re



format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def writeSample(dataset_path, dataset_name):
    """
    get all the samples from sample video and writem them to a tar file using webdataset

    - read in the dataset
    - run the command off the dataset
    - write the samples to a tar file
    """
    datawriter = wds.TarWriter(dataset_path, encoder=False)

    ...


if __name__ == "__main__":
    """
    Run three 
    """
    start = time.time()
    parser = argparse.ArgumentParser(
    description="Perform data preparation for DNN training on a video set.")
    
    parser.add_argument("--dataset_path", type=str, help="Path to the datasets", default=".")
    parser.add_arguement("--dataset-search-string", type=str, help="Grep string to get the datasets", default = "dataset_*.csv")
    
    args = parser.parse_args()
    
    
    command = f"ls {args.dataset_path} | grep {args.dataset_search_string}"
    
