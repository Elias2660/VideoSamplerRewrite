
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
    start_frame, end_frame = row["begin frame"], row["end frame"]

    # max frames in a mp4 is 1000
    if end_frame - start_frame > 1000:
        end_frame = start_frame + 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame_path = f"frames/{video_path}_{frame_num}.jpg"
        cv2.imwrite(frame_path, frame)
        logging.info(f"Saved frame {frame_num} to {frame_path}")
    cap.release()
    return


write_video_to_frames(dataset.iloc[0])


# %%
dataset
# %%
