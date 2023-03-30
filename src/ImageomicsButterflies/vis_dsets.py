from cProfile import label
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from loggers import Logger
from loading_helpers import load_json

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--train_dset", type=str, default="../datasets/high_res_butterfly_data_train.txt")
    parser.add_argument("--test_dset", type=str, default="../datasets/high_res_butterfly_data_test.txt")
    parser.add_argument("--view", type=str, default="D")
    parser.add_argument("--mimic_data", type=str, default="../experiments/mimic_pairs.json")
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")

    args = parser.parse_args()

    return args

def visualize_dataset(train_dset, test_dset, filter_view, mimic_data, logger):
    logger.log(f"Visualizing dataset at: {train_dset} and {test_dset}")
    lbl_name_map = {}
    train_counts = {}
    test_counts = {}
    with open(train_dset, 'r') as f:
        lines = f.readlines()
        for line in lines:
            path, lbl = line.strip().split(" ")
            view = path.split(os.path.sep)[-1].split("_")[1]
            if view != filter_view: continue
            lbl = int(lbl)
            cls_name = os.path.dirname(path).split(os.path.sep)[-1]
            if lbl not in lbl_name_map:
                lbl_name_map[lbl] = cls_name
                test_counts[lbl] = 0
                train_counts[lbl] = 0
            train_counts[lbl] += 1
    with open(test_dset, 'r') as f:
        lines = f.readlines()
        for line in lines:
            path, lbl = line.strip().split(" ")
            view = path.split(os.path.sep)[-1].split("_")[1]
            if view != filter_view: continue
            lbl = int(lbl)
            cls_name = os.path.dirname(path).split(os.path.sep)[-1]
            if lbl not in lbl_name_map:
                lbl_name_map[lbl] = cls_name
                test_counts[lbl] = 0
                train_counts[lbl] = 0
            test_counts[lbl] += 1

    mimic_data = load_json(args.mimic_data)
    order_of_lbls = []
    for pair in mimic_data:
        e_lbl = pair["erato"]["label"]
        m_lbl = pair["melpomene"]["label"]
        if e_lbl not in lbl_name_map:
            lbl_name_map[e_lbl] = pair["erato"]["name"]
            test_counts[e_lbl] = 0
            train_counts[e_lbl] = 0
        if m_lbl not in lbl_name_map:
            lbl_name_map[m_lbl] = pair["melpomene"]["name"]
            test_counts[m_lbl] = 0
            train_counts[m_lbl] = 0
        order_of_lbls.append(e_lbl)
        order_of_lbls.append(m_lbl)
    x_lbls = []
    y_train = []
    y_test = []
    for lbl in order_of_lbls:
        x_lbls.append(lbl_name_map[lbl])
        y_train.append(train_counts[lbl])
        y_test.append(test_counts[lbl])
    
    x = np.arange(len(x_lbls))
    WIDTH = 0.4
    
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.bar(x - WIDTH / 2, y_train, WIDTH, label="Train")
    ax.bar(x + WIDTH / 2, y_test, WIDTH, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(x_lbls, rotation=90)
    #for x, y in zip(range(len(X)), Y):
    #    ax.text(x - 0.3, y + 0.3, y)
    ax.set_xlabel("Subspecies", fontsize=24)
    ax.set_ylabel("Image Count", fontsize=24)
    ax.set_title("Data Count", fontsize=32)
    plt.legend(fontsize=24)
    plt.savefig(os.path.join(logger.get_save_dir(), "class_balance.png"))
    plt.close()
    




if __name__ == "__main__":
    args = get_args()
    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    visualize_dataset(args.train_dset, args.test_dset, args.view, args.mimic_data, logger)

