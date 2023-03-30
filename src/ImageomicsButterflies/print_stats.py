import os
from argparse import ArgumentParser

import numpy as np

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="styleGAN/results/img_to_img/random_default_z/")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    final_img = None
    total = 0
    for root, dirs, files in os.walk(args.exp_dir):
        if "statistics.npz" not in files:
            continue
        total +=1
        pixel_mean = 0
        perceptual_mean = 0
        subspecies = root.split(os.path.sep)[-1]
        statistics = np.load(os.path.join(root, "statistics.npz"))
        pixel_losses = statistics['pixel_losses']
        perceptual_losses = statistics['perceptual_losses']
        
        pixel_mean += pixel_losses[-1]
        perceptual_mean += perceptual_losses[-1]
    #pixel_mean /= total
    #perceptual_mean /= total

    print(f"Pixel mean loss: {pixel_mean}")
    print(f"Perceptual mean loss: {perceptual_mean}")