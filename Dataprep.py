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
import concurrent
import re
import os


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def writeSample(dataset_path, dataset_name, dataset):
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
    sample_end = time.time()
    logging.info(f"Time taken to write the samples for {dataset_name}: {sample_end - sample_start} seconds")
    
    
    datawriter.close()
    # `    frame, video_path, frame_num = frame_data
    #                 base_name = os.path.basename(video_path).replace(' ', '_').replace('.', '_')
    #                 video_time = os.path.basename(video_path).split('.')[0]
    #                 # TODO FIXME Convert the time from the video to the current frame time.
    #                 # TODO Assuming 3fps bee videos
    #                 time_sec = time.mktime(time.strptime(video_time, "%Y-%m-%d %H:%M:%S"))
    #                 time_struct = time.localtime(time_sec + int(frame_num[0]) // 3)
    #                 curtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
    #                 metadata = f"{video_path},{frame_num[0]},{curtime}"
    #                 height, width = frame.size(2), frame.size(3)
    #                 # Now crop to args.width by args.height.
    #                 #ybegin = (height - args.height)//2
    #                 #xbegin = (width - args.width)//2
    #                 #cropped = frame[:,:,ybegin:ybegin+args.height,xbegin:xbegin+args.width]
    #                 # If you would like to debug (and you would like to!) check your images.
    #                 if 1 == args.frames_per_sample:
    #                     if 3 == args.out_channels:
    #                         img = transforms.ToPILImage()(frame[0]/255.0).convert('RGB')
    #                     else:
    #                         img = transforms.ToPILImage()(frame[0]/255.0).convert('L')
    #                     # Now save the image as a png into a buffer in memory
    #                     buf = io.BytesIO()
    #                     img.save(fp=buf, format="png")

    #                     sample = {
    #                         "__key__": '_'.join((base_name, '_'.join(frame_num))),
    #                         "0.png": buf.getbuffer(),
    #                         "cls": row[class_col].encode('utf-8'),
    #                         "metadata.txt": metadata.encode('utf-8')
    #                     }
    #                 else:
    #                     # Save multiple pngs
    #                     buffers = []

    #                     for i in range(args.frames_per_sample):
    #                         if 3 == args.out_channels:
    #                             img = transforms.ToPILImage()(frame[i]/255.0).convert('RGB')
    #                         else:
    #                             img = transforms.ToPILImage()(frame[i]/255.0).convert('L')
    #                         # Now save the image as a png into a buffer in memory
    #                         buffers.append(io.BytesIO())
    #                         img.save(fp=buffers[-1], format="png")

    #                     sample = {
    #                         "__key__": '_'.join((base_name, '_'.join(frame_num))),
    #                         "cls": row[class_col].encode('utf-8'),
    #                         "metadata.txt": metadata.encode('utf-8')
    #                     }
    #                     for i in range(args.frames_per_sample):
    #                         sample[f"{i}.png"] = buffers[i].getbuffer()

    #                 datawriter.write(sample)

    # datawriter.close()`

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
