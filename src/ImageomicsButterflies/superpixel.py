import numpy as np
import cv2 as cv

from argparse import ArgumentParser
from PIL import Image

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--img", type=str, default="/local/scratch/datasets/high_res_butterfly_data_train/aglaope_M/10428112_D_aglaope_M.png")

    return parser.parse_args()

def superpixel(img, iters=6, pixels=40, prior=2, bins=5, levels=4):
    H, W, C = img.shape
    cv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    seeds = cv.ximgproc.createSuperpixelSEEDS(W, H, C, pixels, levels, prior, bins)
    seeds.iterate(cv_img, iters)
    labels = seeds.getLabels()
    return labels

if __name__ == "__main__":
    args = get_args()
    img = np.array(Image.open(args.img))
    superpixel(img)

