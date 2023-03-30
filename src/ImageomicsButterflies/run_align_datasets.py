import os
import shutil
from copy import copy
from argparse import ArgumentParser

import numpy as np
from PIL import Image

from helpers import parse_xlsx_labels

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--org_labels', type=str, default="/research/nfs_chao_209/david/Imagenomics/TableS2HoyalCuthilletal2019-Kozak_curated.xlsx")
    parser.add_argument('--new_labels', type=str, default="/research/nfs_chao_209/david/Imagenomics/TableS2HoyalCuthilletal2019-Original.xlsx")
    parser.add_argument('--org_dset', type=str, default='../datasets/original')
    parser.add_argument('--org_nohybrid_dset', type=str, default='../datasets/original_nohybrid')
    parser.add_argument('--new_dset', type=str, default='../datasets/')
    parser.add_argument('--split', type=float, default=0.8)

    return parser.parse_args()

def get_dataset_splits(dset):
    train_ids = []
    for root, dirs, files in os.walk(os.path.join(dset, "train")):
        for f in files:
            train_ids.append(int(f.split(".")[0].split("_")[0]))
    test_ids = []
    for root, dirs, files in os.walk(os.path.join(dset, "test")):
        for f in files:
            test_ids.append(int(f.split(".")[0].split("_")[0]))

    return {
        "train" : train_ids,
        "test" : test_ids
    }

def align_datasets(control_split, org_split, org_sub_map, split=0.8):
    org_new_split = {
        "train" : copy(control_split["train"]),
        "test" : copy(control_split["test"]),
    }

    remaining_ids = set(org_split["train"] + org_split["test"])
    remaining_ids = remaining_ids.difference(set(org_new_split["train"]))
    remaining_ids = remaining_ids.difference(set(org_new_split["test"]))
    remaining_ids = list(remaining_ids)

    sub_count = { "train" : {}, "test" : {} }
    for sub_id in org_new_split["train"]:
        sub = org_sub_map[sub_id]
        if sub not in sub_count["train"]:
            sub_count["train"][sub] = 0
        sub_count["train"][sub] += 1
    
    for sub_id in org_new_split["test"]:
        sub = org_sub_map[sub_id]
        if sub not in sub_count["test"]:
            sub_count["test"][sub] = 0
        sub_count["test"][sub] += 1

    for rem_id in remaining_ids:
        sub = org_sub_map[rem_id]
        
        cur_splilt = 0.0
        if sub not in sub_count["train"]:
            sub_count["train"][sub] = 0 
            sub_count["test"][sub] = 0
        else: 
            cur_splilt = sub_count["train"][sub] / (sub_count["train"][sub] + sub_count["test"][sub])
        
        t = "test"
        if cur_splilt <= split:
            t = "train"
        sub_count[t][sub] += 1
        org_new_split[t].append(rem_id)

    return org_new_split
        
def create_new_dataset(new_split, sub_map, dset_root):
    new_dset_root = dset_root + "_new"

    t = "train"
    out_dir = os.path.join(new_dset_root, t)
    for sub_id in new_split[t]:
        sub = sub_map[sub_id]
        sub_dir = os.path.join(out_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        fname = f"{sub_id}_D_lowres.png"
        src_path = os.path.join(dset_root, "train", sub, fname)
        if not os.path.exists(src_path):
            src_path = os.path.join(dset_root, "test", sub, fname)

        shutil.copy(src_path, os.path.join(sub_dir, fname))

    t = "test"
    out_dir = os.path.join(new_dset_root, t)
    for sub_id in new_split[t]:
        sub = sub_map[sub_id]
        sub_dir = os.path.join(out_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        fname = f"{sub_id}_D_lowres.png"
        src_path = os.path.join(dset_root, "train", sub, fname)
        if not os.path.exists(src_path):
            src_path = os.path.join(dset_root, "test", sub, fname)

        shutil.copy(src_path, os.path.join(sub_dir, fname))



if __name__ == "__main__":
    args = get_args()
    
    org_sub_map, org_hybrid_map = parse_xlsx_labels(args.org_labels, return_hybrid=True)
    new_sub_map, new_hybrid_map = parse_xlsx_labels(args.new_labels, return_hybrid=True)

    org_split = get_dataset_splits(args.org_dset) 
    org_nohybrid_split = get_dataset_splits(args.org_nohybrid_dset) 
    new_split = get_dataset_splits(args.new_dset) 

    #! New will be our base. No modification will be done to the new split.
    
    #? Do both the original datasets align with the train/test split of the new dataset?
    # In other words, do their train splits contain every image in the new dataset train split (same for testing).

    print(len(set(new_split["train"]).difference(set(org_nohybrid_split["train"]))))
    print(len(set(new_split["test"]).difference(set(org_nohybrid_split["test"]))))

    print(len(set(new_split["train"]).difference(set(org_split["train"]))))
    print(len(set(new_split["test"]).difference(set(org_split["test"]))))

    # Output: 51, 67, 54, 69
    # Answer: No

    exit() #! Comment this out to make this program actually create new datasets

    # Realign datasets
    org_nohybrid_split_new = align_datasets(new_split, org_nohybrid_split, org_sub_map, args.split)
    assert (len(org_nohybrid_split["train"]) + len(org_nohybrid_split["test"])) == (len(org_nohybrid_split_new["train"]) + len(org_nohybrid_split_new["test"])), "Must not add or remove from original dataset"
    
    org_split_new = align_datasets(org_nohybrid_split_new, org_split, org_sub_map, args.split)
    assert (len(org_split["train"]) + len(org_split["test"])) == (len(org_split_new["train"]) + len(org_split_new["test"])), "Must not add or remove from original dataset"

    create_new_dataset(org_nohybrid_split_new, org_sub_map, args.org_nohybrid_dset)
    create_new_dataset(org_split_new, org_sub_map, args.org_dset)