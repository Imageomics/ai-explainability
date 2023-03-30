from argparse import ArgumentParser

import imageio
import numpy as np

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--projections", type=str, default="styleGAN/results/img_to_img/random_default_z/aglaope_M/projections.npz")
    parser.add_argument("--img_idx", type=int, default=0)
    parser.add_argument("--out_file", type=str, default="out.mp4")

    args = parser.parse_args()
    return args

def load_projections(path):
    return np.transpose(np.load(path)['projections'], axes=[0, 1, 3, 4, 2])

def create_video(frames, out_dest):
    video = imageio.get_writer(out_dest, mode='I', fps=30, codec='libx264')
    for frame in frames:
        video.append_data((frame * 255).astype(np.uint8))
    video.close()
if __name__ == "__main__":
    args = get_args()
    projections = load_projections(args.projections)
    frames = projections[:, args.img_idx]
    create_video(frames, args.out_file)

    