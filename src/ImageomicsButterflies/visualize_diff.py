import os
from pydoc import plain

import numpy as np

from argparse import ArgumentParser
from PIL import Image

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--thresh", type=float, default=0.0)
    parser.add_argument("--start_img", type=int, default=0)
    parser.add_argument("--end_img", type=int, default=499)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    start_img = np.array(Image.open(os.path.join(args.root, f"img_frames/{args.start_img}.png"))) / 255
    end_img = np.array(Image.open(os.path.join(args.root, f"img_frames/{args.end_img}.png"))) / 255

    plain_diff = np.abs(end_img - start_img).sum(2)
    plain_diff[plain_diff > 1.0] = 1.0
    plain_diff[plain_diff < args.thresh] = 0.0
    print(plain_diff.shape)
    img = Image.fromarray((plain_diff * 255).astype(np.uint8))
    img.save("vis_plain_diff.png")