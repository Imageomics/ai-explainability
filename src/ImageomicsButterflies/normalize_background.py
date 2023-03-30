import os
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from loggers import Logger

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../datasets/high_res_butterfly_data_test.txt")
    parser.add_argument("--output", type=str, default="/local/scratch/datasets/high_res_butterfly_data_test_norm")
    parser.add_argument("--exp_name", type=str, default="debug")

    args = parser.parse_args()
    assert args.dataset is not None, "Must provide a dataset"

    return args

def contour_mask(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros_like(gray).astype(np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    return masked

def dist(a, b):
    return np.sqrt(((a - b) ** 2).sum())

def region_growing_mask(path, thresh=5):
    img = np.array(Image.open(path))
    h, w = img.shape[:2]
    visited = ["0_0", f"0_{w-1}", f"{h-1}_{w-1}", f"{h-1}_0"]
    queue = [(0, 0, img[0, 0]), (0, w-1, img[0, w-1]), (h-1, w-1, img[h-1, w-1]), (h-1, 0, img[h-1, 0])]
    background = np.ones_like(img[:, :, 0]).astype(np.uint8)
    background[0,0] = 0
    background[0,w-1] = 0
    background[h-1,w-1] = 0
    background[h-1,0] = 0
    i = 0

    def get_neighbor_points(row, col):
        points = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0: continue
                points.append((max(min(row + y, img.shape[0]-1), 0), max(min(col + x, img.shape[1]-1), 0)))
        return points
    while len(queue) > 0:
        i += 1
        if i % 1000 == 0:
            Image.fromarray(background * 255).save("in_progress_mask.png")
        row, col, color = queue.pop(0)
        points = get_neighbor_points(row, col)
        for y, x in points:
            c = img[y, x]
            d = dist(color, c)
            if d < thresh:
                background[y, x] = 0
                if f"{y}_{x}" not in visited:
                    queue.append((y, x, c))
                    visited.append(f"{y}_{x}")

    return background

def threshold_mask(path, thresh=5):
    img = np.array(Image.open(path))
    px = img[0, 0]
    print(px)
    diff = np.sqrt(((img - px)**2).sum(2))
    mask = np.logical_and(diff > -thresh, diff < thresh)
    print(mask[0][0])
    mask = np.logical_not(mask).astype(np.uint8)
    return mask

def normalize_background(dataset, output):
    os.makedirs(output, exist_ok=False)

    lbl_name_map = {}
    lbl_paths_map = {}
    with open(dataset, 'r') as f:
        lines = f.readlines()
        for line in lines:
            path, lbl = line.strip().split(" ")
            if path.split(os.path.sep)[-1].split("_")[1] != "D": continue
            lbl = int(lbl)
            cls_name = os.path.dirname(path).split(os.path.sep)[-1]
            if lbl not in lbl_name_map:
                lbl_name_map[lbl] = cls_name
                os.makedirs(os.path.join(output, cls_name))
                lbl_paths_map[lbl] = []
            lbl_paths_map[lbl].append(path)

    for lbl in lbl_paths_map:
        avg_color = np.zeros(3)
        for path in lbl_paths_map[lbl]:
            fname = path.split(os.path.sep)[-1]
            start_img = Image.open(path)
            start_img.save("before_normalize.png")
            #mask = contour_mask(path)
            #mask = threshold_mask(path)
            mask = region_growing_mask(path)
            new_image = np.array(start_img)
            new_image[mask == 0] = np.array([210, 210, 210])
            Image.fromarray(mask).save("mask.png")
            Image.fromarray(new_image).save("after_normalize.png")
            Image.fromarray(new_image).save(os.path.join(output, lbl_name_map[lbl], fname))

            avg_color += np.array(new_image)[0,0,:]
        avg_color /= len(lbl_paths_map[lbl])
        print(f"Avg Color for {lbl_name_map[lbl]}: {avg_color}")

if __name__ == "__main__":
    args = get_args()
    normalize_background(args.dataset, args.output)

