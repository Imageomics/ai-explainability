import os
import cv2
import numpy as np

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../output/debug")
    parser.add_argument("--img_dir", type=str, default="imgs")
    parser.add_argument("--img_key", type=str, default="transform_results")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    img_paths = []
    for root, dirs, paths in os.walk(os.path.join(args.log_dir, args.img_dir)):
        tmp = list(filter(lambda x: args.img_key in x, paths))
        img_paths.extend(tmp)
    img_paths = sorted(img_paths, key=lambda x: float(x.split("_")[0]))
    img = cv2.imread(os.path.join(args.log_dir, args.img_dir, img_paths[0]))
    print(img.shape)
    frame_size = (img.shape[1], img.shape[0])
    out = cv2.VideoWriter(os.path.join(args.log_dir, 'transform_video.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 10, frame_size)
    for p in img_paths:
        img = cv2.imread(os.path.join(args.log_dir, args.img_dir, p))
        img = cv2.resize(img, frame_size)
        out.write(img)
    cv2.destroyAllWindows()
    out.release()