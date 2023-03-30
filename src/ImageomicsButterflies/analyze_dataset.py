import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from loggers import Logger

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../datasets/high_res_butterfly_data_train.txt")
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")

    args = parser.parse_args()
    assert args.dataset is not None, "Must provide a dataset"

    return args

#! FILTERS FOR VIEW Dorsal
def analyze_background_color(dataset, logger):
    logger.log(f"Visualizing dataset at: {dataset}")

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
                lbl_paths_map[lbl] = []
            lbl_paths_map[lbl].append(path)

    background_pixels = [[] for _ in range(max(lbl_paths_map.keys())+1)]
    for lbl in lbl_paths_map:
        for path in lbl_paths_map[lbl]:
            px = np.array(Image.open(path))[0,0,:]
            background_pixels[lbl].append(px)

    
    within_subspecies_mean = []
    within_subspecies_var = []
    min_mean = 999
    min_mean_name = ""
    max_mean = 0
    max_mean_name = ""
    for lbl, subspecies_pxs in enumerate(background_pixels):
        mean = np.array(subspecies_pxs).mean(0)
        var = np.array(subspecies_pxs).var(0)
        if max(mean) > max_mean:
            max_mean = max(mean)
            max_mean_name = lbl_name_map[lbl]
        if min(mean) < min_mean:
            min_mean = min(mean)
            min_mean_name = lbl_name_map[lbl]
        #print(f"{lbl_name_map[lbl]} mean: {mean}")
        #print(f"{lbl_name_map[lbl]} variance: {var}")
        within_subspecies_mean.append(mean)
        within_subspecies_var.append(var)

    print(f"Max: {max_mean_name} {max_mean}")
    print(f"Min: {min_mean_name} {min_mean}")

    within_subspecies_mean = np.array(within_subspecies_mean) 
    within_subspecies_var = np.array(within_subspecies_var) 
    print(f"Average variance within subspecies: {within_subspecies_var.mean(0)}")
    print(f"Variance of subspecies mean: {within_subspecies_mean.var(0)}")


def visualize_dataset(dataset, logger):
    logger.log(f"Visualizing dataset at: {dataset}")
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
                lbl_paths_map[lbl] = []
            lbl_paths_map[lbl].append(path)
    
    X = []
    Y = []
    for lbl in lbl_paths_map:
        X.append(lbl_name_map[lbl])
        Y.append(len(lbl_paths_map[lbl]))
    
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.bar(range(len(X)), Y)
    ax.set_xticks(range(len(X)))
    ax.set_xticklabels(X, rotation=90)
    for x, y in zip(range(len(X)), Y):
        ax.text(x - 0.3, y + 0.3, y)
    ax.set_xlabel("Subspecies")
    ax.set_ylabel("Image Count")
    ax.set_title("Class Balance")
    plt.savefig(os.path.join(logger.get_save_dir(), "class_balance.png"))
    plt.close()
    




if __name__ == "__main__":
    args = get_args()
    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    #visualize_dataset(args.dataset, logger)
    analyze_background_color(args.dataset, logger)

