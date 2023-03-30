import os
import random
from argparse import ArgumentParser
from turtle import color

from tqdm import tqdm 

import numpy as np
import openpyxl

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt

from PIL import Image

from loggers import Logger
from datasets import ImageFolder

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--min_loss", type=float, default=0.1)
    parser.add_argument("--augment_strength", type=float, default=0.1)
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--train_dataset", type=str, default="../datasets/train")
    parser.add_argument("--test_dataset", type=str, default="../datasets/test")
    parser.add_argument("--view", type=str, default="D")
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--excel_file", type=str, default="/research/nfs_chao_209/david/Imagenomics/TableS2HoyalCuthilletal2019-Kozak_curated.xlsx")


    args = parser.parse_args()
    args.gpus = ",".join(map(lambda x: str(x), args.gpus))
    return args

def setup(args):
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def parse_excel(path):
    wb = openpyxl.load_workbook(path)
    ws = wb['Table S2']
    nrows = ws.max_row
    data = []
    for i, row in enumerate(ws.rows):
        if i <= 1: continue
        data_row = []
        for j, cell in enumerate(row):
            if j in [0, 2, 4, 8]:
                data_row.append(cell.value)
        data.append(data_row)
    return data

def transform_data(data):
    subspecies = set()
    for row in data:
        subspecies.add(row[2])
    subspecies = sorted(list(subspecies))
    sub_to_lbl_map = {}
    lbl_to_sub_map = {}
    for i, sub in enumerate(subspecies):
        sub_to_lbl_map[sub] = i
        lbl_to_sub_map[i] = sub

    def view_to_num(x):
        return 1 if "dorsal" in x else 0
    def hybrid_to_num(x):
        return 1 if "hybrid" in x else 0
    rv = list(map(lambda x: [sub_to_lbl_map[x[2]], view_to_num(x[1]), hybrid_to_num(x[3])], data))
    return np.array(rv), lbl_to_sub_map

def save_results(processed_data, lbl_to_sub_map, logger):
    num_lbls = processed_data[:, 0].max() + 1
    dorsal_mask = processed_data[:, 1] == 1
    hybrid_mask = processed_data[:, 2] == 1

    total_count = len(processed_data)
    total_dorsal = dorsal_mask.sum()
    total_ventral = total_count - total_dorsal
    total_hybrid = hybrid_mask.sum()
    total_non_hybrid = total_count - total_hybrid

    total_hybrid_dorsal = np.logical_and(hybrid_mask, dorsal_mask).sum()
    total_hybrid_ventral = total_hybrid - total_hybrid_dorsal
    total_non_hybrid_dorsal = np.logical_and(np.logical_not(hybrid_mask), dorsal_mask).sum()
    total_non_hybrid_ventral = total_non_hybrid - total_non_hybrid_dorsal

    top_data = [total_hybrid, total_ventral, total_non_hybrid_ventral, total_hybrid_ventral]
    top_labels = ["hybrid", "ventral", "ventral", "ventral"]
    color_top = ["blue", "orange", "orange", "orange"]
    bottom_labels = ["non-hybrid", "dorsal", "dorsal", "dorsal"]
    color_bottom = ["green", "red", "red", "red"]
    bottom_data = [total_non_hybrid, total_dorsal, total_non_hybrid_dorsal, total_hybrid_dorsal]
    labels = ["hybrid v. non-hybrid", "dorsal v. ventral", "non-hybrid (D v. V)", "hybrid (D v. V)"]


    fig, ax = plt.subplots()
    WIDTH = 0.35
    for i in range(len(top_data)):
        bar_bottom = ax.bar(i, bottom_data[i], WIDTH, label=bottom_labels[i], color=color_bottom[i])
        bar_top = ax.bar(i, top_data[i], WIDTH, bottom=bottom_data[i], label=top_labels[i], color=color_top[i])
        ax.bar_label(bar_bottom, label_type='center')
        ax.bar_label(bar_top, label_type='center')
        ax.bar_label(bar_top)
    ax.set_ylabel('Image Count')
    ax.set_title('Butterfly Dataset Analysis')
    ax.set_xticks(range(len(top_data)), labels=labels)
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.legend()

    plt.savefig(os.path.join(logger.get_save_dir(), "data_count.png"))
    plt.close()


if __name__ == "__main__":
    args = get_args()
    setup(args)

    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    # Save Args
    logger.save_json(args.__dict__, "args.json")
    #train_dset = ImageFolder(args.train_dataset, transform=None)
    #test_dset = ImageFolder(args.test_dataset, transform=None)

    data = parse_excel(args.excel_file)
    processed_data, lbl_to_sub_map = transform_data(data)
    save_results(processed_data, lbl_to_sub_map, logger)

