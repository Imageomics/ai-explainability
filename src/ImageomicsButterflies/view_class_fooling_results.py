from json import load
import os
from argparse import ArgumentParser

import numpy as np
from loading_helpers import load_json

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="/research/nfs_chao_209/david/class_fooling/autoencoder_default_w_w/")
    parser.add_argument("--mimic_data", type=str, default="../experiments/mimic_pairs_filtered.json")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    mimic_data = load_json(args.mimic_data)
    name_to_lbl_map = {}
    for pair in mimic_data:
        name_to_lbl_map[pair["erato"]["name"]] = pair["erato"]["label"]
        name_to_lbl_map[pair["melpomene"]["name"]] = pair["melpomene"]["label"]

    final_img = None
    total = 0
    conf_total = 0
    min_total = 0
    for root, dirs, files in os.walk(args.exp_dir):
        if "statistics.npz" not in files:
            continue
        subspecies = root.split(os.path.sep)[-1].split("_")[-2]
        #print(subspecies)
        total +=1
        avg_conf = 0.0
        avg_min_loss = 0
        lbl = name_to_lbl_map[subspecies]
        #print(lbl)
        subspecies = root.split(os.path.sep)[-1]
        statistics = np.load(os.path.join(root, "statistics.npz"))
        confidences = statistics['image_confs']
        min_losses = statistics['min_losses']
        conf_total += confidences[-1][:, lbl].mean()
        min_total += min_losses[-1]

    conf_total /= total
    min_total /= total

    print(f"Target Conf mean: {conf_total}")
    print(f"Min mean loss: {min_total}")