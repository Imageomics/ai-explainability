from argparse import ArgumentParser

import numpy as np
import PIL.Image as Image

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--projections", type=str, default="styleGAN/results/img_to_img/random_default_z/rosina_M/projections.npz")
    parser.add_argument("--img_num", type=int, default=0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    projections = np.load(args.projections)['projections']
    projection = (np.transpose(projections[-1][args.img_num], axes=[1, 2, 0]) * 255).astype(np.uint8)
    Image.fromarray(projection).save("test.png")
