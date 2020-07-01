import io
import os
import cv2
import time
import zipfile
import argparse
import numpy as np 
import pandas as pd
import concurrent.futures
from decouple import config


DATA_PATH = config("FACE_DATA")+"face_landmarks1/"
files = os.listdir(DATA_PATH)
print("files in data directory: ", files)

def draw_batch(batch):
    rows = []
    BATCH_LENGTH = len(batch)
    for yi in range(0, BATCH_LENGTH-1, 8):
        cols = []
        row = batch[yi:yi+8]
        for data in row:
            img = data["image"]
            keypoints = data["keypoints"]
            img = plot_keypoints(img, keypoints)
            cols.append(img)
        row = np.hstack(cols)
        rows.append(row)
    batch = np.vstack(rows)
    cv2.imshow("batch", batch)
    cv2.waitKey(0)
    cv2.destroyWindow("batch")


def process_img(data_row, root_path=DATA_PATH+"training/", size=96):
    index, data = data_row
    image_name = data.iloc[0]
    img_path = root_path + image_name
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = data.iloc[1:].values.reshape(-1, 2)

    scale = 1
    h, w = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim 
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        keypoints = keypoints * scale
    else:
        new_h, new_w = h, w
    
    start_x = size//2 - new_w // 2
    start_y = size//2 - new_h // 2
    keypoints += [start_x, start_y] 
    resized_img = np.zeros((size, size)).astype("uint8")
    resized_img[start_y:start_y+new_h, start_x:start_x+new_w] = img
    return {"name" : image_name, "image": resized_img, "keypoints" : keypoints}


def plot_keypoints(img, keypoints):
    if len(img.shape) == 2:
        img = np.dstack([img]*3).astype("uint8")
    for (x, y) in keypoints:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    return img


if __name__ == "__main__":
    EXTRACTION_PATH = "../data/landmarks_dataset.npz"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-S", "--state", required=True, type=str,
        help="extract/viusalize extract dataset from source or visualize saved data"
    )
    parser.add_argument(
        "-s", "--save", required=False, action="store_true",
        help="saveing extracted dataset from source to project root data dir"
    )

    extracted_data = None
    args = parser.parse_args()
    if args.state == "extract":
        dataset = pd.read_csv(DATA_PATH+"training_frames_keypoints.csv")
        print("---> columns length: ", len(dataset.columns))
        
        t1 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executer:
            result = executer.map(process_img, dataset.iterrows())
        t2 = time.perf_counter()
        print(f"time elapsed: {t2 - t1: .2f}")

        extracted_data = list(result)
        if args.save:
            np.savez_compressed(EXTRACTION_PATH, extracted_data)
            print("saved extracted data in ", EXTRACTION_PATH)
    elif args.state == "visualize":
        if os.path.exists(EXTRACTION_PATH):
            extracted_data = np.load(EXTRACTION_PATH, allow_pickle=True)
            extracted_data = extracted_data["arr_0"]
            batch = extracted_data[:64]
            draw_batch(batch)

   