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


DATA_PATH = config("FACE_DATA")+"facial_expressions1/Training/"
files = os.listdir(DATA_PATH)
print("files in data directory: ", files)

def draw_batch(batch):
    rows = []
    BATCH_LENGTH = len(batch)
    for yi in range(0, BATCH_LENGTH-1, 8):
        cols = []
        row = batch[yi:yi+8]
        for data in row:
            img = np.dstack([data["image"]]*3)
            label = data["label"]
            cv2.rectangle(img, (0, 0), (35, 10), (70, 0, 240), -1)
            cv2.putText(img, label, (5, 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
            cols.append(img)
        row = np.hstack(cols)
        rows.append(row)
    batch = np.vstack(rows)
    batch = cv2.resize(batch, None, fx=2, fy=2)
    cv2.imshow("batch", batch)
    cv2.waitKey(0)
    cv2.destroyWindow("batch")


def process_img(img_path, size=48):
    # image_name = os.path.basename(img_path)
    label = os.path.dirname(img_path).split("/")[-1]
    image_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1
    h, w = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim 
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
    else:
        new_h, new_w = h, w
    
    start_x = size//2 - new_w // 2
    start_y = size//2 - new_h // 2
    resized_img = np.zeros((size, size)).astype("uint8")
    resized_img[start_y:start_y+new_h, start_x:start_x+new_w] = img
    return {"name" : image_name, "image": resized_img, "label" : label}


if __name__ == "__main__":
    EXTRACTION_PATH = "../data/facial_expressions_dataset.npz"
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
    sub_dirs = {}
    if args.state == "extract":
        for sub_dir in os.listdir((DATA_PATH)):
            content = []
            for img_name in os.listdir(DATA_PATH+sub_dir):
                content.append(DATA_PATH + sub_dir + "/" + img_name)
            sub_dirs[sub_dir] = content
        t1 = time.perf_counter()
        extracted_data = []
        for sub_dir in sub_dirs:
            with concurrent.futures.ThreadPoolExecutor() as executer:
                result = executer.map(process_img, sub_dirs[sub_dir])
                extracted_data.extend(list(result))
        t2 = time.perf_counter()
        print(f"time elapsed: {t2 - t1: .2f}")
        # np.random.shuffle(extracted_data)
        # for data in extracted_data:
        #     print("name: ", data["name"], "image: ", data["image"].shape, "label: ", data["label"])

        if args.save:
            np.savez_compressed(EXTRACTION_PATH, extracted_data)
            print("saved extracted data in ", EXTRACTION_PATH)
    elif args.state == "visualize":
        if os.path.exists(EXTRACTION_PATH):
            extracted_data = np.load(EXTRACTION_PATH, allow_pickle=True)
            extracted_data = extracted_data["arr_0"]
            np.random.shuffle(extracted_data)
            batch = extracted_data[:64]
            draw_batch(batch)

   