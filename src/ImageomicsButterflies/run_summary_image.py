import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
from PIL import Image

from loading_helpers import load_json

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--verbose',              help='print all messages?', action="store_true", default=False)
    parser.add_argument('--overwrite',              help='overwrite results', action="store_true", default=False)
    parser.add_argument('--outdir_root', type=str, default="/research/nfs_chao_209/david/")
    parser.add_argument('--experiments_path', type=str, default="../experiments/summary_image.json")
    parser.add_argument('--mimic_pairs_path', type=str, default="../experiments/mimic_pairs_filtered.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Setup
    args = get_args()

    # Load data
    experiments = load_json(args.experiments_path)
    mimic_pairs = load_json(args.mimic_pairs_path)

    pair_data = []
    for mimic_pair in mimic_pairs:
            melpomene = f"{mimic_pair['melpomene']['name']}_M"
            erato = f"{mimic_pair['erato']['name']}_E"

            pair_data.append(f"{melpomene}_to_{erato}")
            pair_data.append(f"{erato}_to_{melpomene}")

    results_dir = os.path.join(args.outdir_root, "summary_images")
    os.makedirs(results_dir, exist_ok=args.overwrite)
    for exp_i, exp in enumerate(experiments):
        exp_start_time = perf_counter()
        visualization_dir = exp["visualization_dir"]
        projection_name = exp["projection_name"]
        exp_outdir = os.path.join(results_dir, projection_name)
        confidence_visualize = exp["confidence"]
        thresh = exp["thresh"]
        #use_median_filter = exp["median_filter"]
        buckets = exp["buckets"]
        colorspaces = exp["colorspaces"]
        img_idx = exp["index"]

        os.makedirs(exp_outdir, exist_ok=args.overwrite)
        for pair in pair_data:
            start_img = None
            end_img = None
            summary_img = None
            for bucket in buckets:
                row = None
                if start_img is not None:
                    row = np.copy(start_img)
                for colorspace in colorspaces:
                    dir_name = f"{projection_name}_{colorspace}"
                    if confidence_visualize:
                        dir_name += "_conf"
                    dir_name += f"_buckets_{bucket}_{thresh}"
                    target_dir = os.path.join(visualization_dir, dir_name, pair)
                    if start_img is None:
                        start_img = np.array(Image.open(os.path.join(target_dir, f"start_{img_idx}.png")))
                        row = np.copy(start_img)
                    if end_img is None:
                        end_img = np.array(Image.open(os.path.join(target_dir, f"end_{img_idx}.png")))
                    img = np.array(Image.open(os.path.join(target_dir, f"start_all_{img_idx}.png")))
                    row = np.concatenate((row, img), axis=1)
                row = np.concatenate((row, end_img), axis=1)
                if summary_img is None:
                    summary_img = np.copy(row)
                else:
                    summary_img = np.concatenate((summary_img, row), axis=0)
            
            Image.fromarray(summary_img).save(os.path.join(exp_outdir, f"{pair}_{img_idx}.png"))



