import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(img, size, **args):
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
    return resized_img


def load_data(data_path):
    dataset = np.load(data_path, allow_pickle=True)
    print(dataset["arr_0"].size)
    return #{"name" : image_name, "image": resized_img, "keypoints" : keypoints}



if __name__ == "__main__":
    landmark_dataset_path = "../data/landmarks_dataset.npz"
    load_data(landmark_dataset_path)