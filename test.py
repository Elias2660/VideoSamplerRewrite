
#%%
import cv2
import numpy as np
import pandas as pd


#%%
cap = cv2.VideoCapture("2024-07-03 12:27:53.850968.mp4")
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

for i in range(1000):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    
    
cv2.destroyAllWindows()
cap.release()
# %%


