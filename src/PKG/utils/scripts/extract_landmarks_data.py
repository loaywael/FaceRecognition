import io
import os
import cv2
import zipfile
import numpy as np 
import pandas as pd
from decouple import config
import matplotlib.pyplot as plt


DATA_PATH = config("FACE_DATA")
files = os.listdir(DATA_PATH)

def draw_batch(batch, marks):
    rows = []
    for yi in range(len(batch)//8):
        cols = []
        for xi in range(len(batch)//8):
            img = np.dstack([batch[xi]]*3).astype("uint8")
            mark = marks.iloc[xi, :].astype("int64")
            nCols = len(marks.columns)
            for i in range(nCols-1):
                pt = tuple(mark[i:i+2])
                cv2.circle(img, pt, 2, (0, 255, 0), -1)
            cols.append(img)
        row = np.hstack(cols)
        rows.append(row)
    batch = np.vstack(rows)
    cv2.imshow("batch", batch)
    cv2.waitKey(0)
    cv2.destroyWindow("batch")


with zipfile.ZipFile(DATA_PATH+files[0], "r") as zf:
    zf.printdir()
    landmarks = pd.read_csv(zf.open("facial_keypoints.csv", "r"))
    with io.BufferedReader(zf.open("face_images.npz", "r")) as npf:
        data = np.load(npf)
        print(data.files)
        # print(data.f.face_images.shape)
        data = data["face_images"]
        data = np.moveaxis(data, -1, 0)
    draw_batch(data[:64], landmarks[:64])

cv2.destroyAllWindows()
