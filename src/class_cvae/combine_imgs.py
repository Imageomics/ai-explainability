from argparse import ArgumentParser

import numpy as np

from PIL import Image

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--imgs', nargs='+', type=str, default=[
        "tmp/exp2_1.png",
        "tmp/exp3.png",
        "tmp/exp4.png",
        "tmp/exp5.png",
        "tmp/exp6_latent.png",
        "tmp/exp6_img.png",
        "tmp/exp7_latent.png",
        "tmp/exp7_img.png"
    ])
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    final = None
    for img_path in args.imgs:
        img = Image.open(img_path)
        img_arr = np.array(img)
        if final is None:
            final = img_arr
        else:
            final = np.concatenate((final, img_arr), axis=1)
    
    Image.fromarray(final).save("tmp/combined.png")
