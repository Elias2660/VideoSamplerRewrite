
# %%

# open the dataset_0.csv file 

import pandas as pd
import logging
import cv2

# %%

# open the dataset_0.csv file
dataset = pd.read_csv("dataset_0.csv")

# for each row, open the mp4 file with open cv2 

def write_video_to_frames(row):
    video_path = row["file"]
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    start_frame, end_frame = int(row[" begin frame"]), int(row[" end frame"])
    # max frames in a mp4 is 1000
    if end_frame - start_frame > 1000:
        end_frame = start_frame + 1000

    # write the frames as individual images to the frames directory 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num >= start_frame and frame_num < end_frame:
            cv2.imwrite(f"frames/{frame_num}.jpg", frame)
        frame_num += 1
    cap.release()
    return


write_video_to_frames(dataset.iloc[0])

