import os
import random

import torch
import numpy as np
import openpyxl

# Set CUDA device
def cuda_setup(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

# Set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Get labels
def parse_xlsx_labels(path="/research/nfs_chao_209/david/Imagenomics/TableS2HoyalCuthilletal2019-Kozak_curated.xlsx", return_hybrid=False):
    wb = openpyxl.load_workbook(path)
    ws = wb['Table S2']
    nrows = ws.max_row
    id_to_subspecies_map = {}
    is_hybrid = {}
    for i, row in enumerate(ws.rows):
        if i <= 1: continue
        id_num, subspecies, hybrid = None, None, None
        for j, cell in enumerate(row):
            if j == 0:
                id_num = cell.value
            elif j == 4:
                subspecies = cell.value
            elif j == 8:
                hybrid = cell.value == "hybrid"
            elif j > 8:
                break
        assert id_num is not None
        assert subspecies is not None
        id_to_subspecies_map[id_num] = subspecies
        is_hybrid[id_num] = hybrid
    if return_hybrid:
        return id_to_subspecies_map, is_hybrid
    return id_to_subspecies_map
