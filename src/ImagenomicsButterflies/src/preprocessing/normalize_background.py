import os
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image, ImageEnhance

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/local/scratch/datasets/butterflies")
    parser.add_argument("--output", type=str, default="/local/scratch/datasets/butterflies_norm")
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
    diff = a - b
    print(diff)
    return np.sqrt((diff ** 2).sum())

def region_growing_mask(path, thresh=3):
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

def normalize_background(dataset, output, query_point = [0.0, 0.5]):
    os.makedirs(output, exist_ok=True)
    lines = []
    for root, dirs, files in os.walk(dataset):
        for f in files:
            path = os.path.join(root, f)
            lines.append(path)
    
    tmp_img = np.array(Image.open(lines[0]))
    q_row = int(query_point[0] * tmp_img.shape[0]-1)
    q_col = int(query_point[1] * tmp_img.shape[1]-1)
    colors = []
    for path in lines:
        img = np.array(Image.open(path))
        colors.append(img[q_row, q_col])

    norm_color = np.array(colors).mean(0).astype(np.uint8)
    print(norm_color)
    
    for path in lines:
        img = Image.open(path)
        img_filter = ImageEnhance.Brightness(img)
        best_val = 1.0
        short_dist = 1000
        for i in range(301):
            val = i / 100
            q_img = img_filter.enhance(val)
            d = dist(np.array(q_img)[q_row, q_col], norm_color) #TODO: dist function seems broken
            print(d, short_dist)
            print(np.array(q_img)[q_row, q_col])
            print(norm_color)
            if d < short_dist:
                print("CHANGE")
                short_dist = d
                best_val = val

        final_img = img_filter.enhance(best_val)
        print(np.array(final_img)[q_row, q_col])
        exit()

if __name__ == "__main__":
    args = get_args()
    normalize_background(args.dataset, args.output)

