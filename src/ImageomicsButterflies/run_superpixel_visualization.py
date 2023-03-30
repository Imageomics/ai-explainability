import os
from argparse import ArgumentParser
from time import perf_counter

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.colors as mcolors
from PIL import Image
from torchvision import transforms

from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from superpixel import superpixel

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--darken_factor", type=float, default=1.5)
    parser.add_argument('--verbose',              help='print all messages?', action="store_true", default=False)
    parser.add_argument('--overwrite',              help='overwrite results', action="store_true", default=False)
    parser.add_argument('--outdir_root', type=str, default="/research/nfs_chao_209/david/")
    parser.add_argument('--experiments_path', type=str, default="../experiments/superpixel_visualization.json")
    parser.add_argument('--mimic_pairs_path', type=str, default="../experiments/mimic_pairs_filtered.json")
    parser.add_argument('--dataset_root', type=str, default="../datasets/high_res_butterfly_data_test/")
    parser.add_argument('--backbone', type=str, default="../saved_models/vgg_backbone_nohybrid_D_norm.pt")
    parser.add_argument('--classifier', type=str, default="../saved_models/vgg_classifier_nohybrid_D_norm.pt")

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def save_img(x, path):
    Image.fromarray((x * 255).astype(np.uint8)).save(path)

def convert_to_input(x):
    img = torch.from_numpy(np.transpose(x, axes=[2, 0, 1])).unsqueeze(0).cuda()
    return NORMALIZE(img)

if __name__ == "__main__":
    # Time
    all_start_time = perf_counter()

    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)

    # Load models
    _, _, F, C = load_models(None, args.backbone, args.classifier)
    sm = nn.Softmax(dim=1)

    # Load data
    experiments = load_json(args.experiments_path)
    mimic_pairs = load_json(args.mimic_pairs_path)

    pair_data = []
    for mimic_pair in mimic_pairs:
            erato_path = os.path.join(args.dataset_root, f"{mimic_pair['erato']['name']}_E")
            assert os.path.exists(erato_path), f"{erato_path} does not exists"
            melpomene_path = os.path.join(args.dataset_root, f"{mimic_pair['melpomene']['name']}_M")
            assert os.path.exists(melpomene_path), f"{melpomene_path} does not exists"

            #! Notice we swap the labels for fooling the classifier
            melpomene = melpomene_path.split(os.path.sep)[-1]
            melpomene_lbl = mimic_pair['melpomene']['label']

            erato = erato_path.split(os.path.sep)[-1]
            erato_lbl = mimic_pair['erato']['label']

            pair_data.append((f"{melpomene}_to_{erato}", erato_lbl))
            pair_data.append((f"{erato}_to_{melpomene}", melpomene_lbl))

    results_dir = os.path.join(args.outdir_root, "superpixel_visualization")
    os.makedirs(results_dir, exist_ok=args.overwrite)
    save_json(args.__dict__, os.path.join(results_dir, "args.json"))
    for exp_i, exp in enumerate(experiments):
        exp_start_time = perf_counter()
        projection_path = exp["projection_folder"]
        
        exp_name_root = "/".join(projection_path.split(os.path.sep)[-2:])

        exp_outdir = os.path.join(results_dir, exp_name_root)
        os.makedirs(exp_outdir, exist_ok=args.overwrite)
        save_json(exp, os.path.join(exp_outdir, f"exp_args.json"))

        for cls_fool_i, (cls_fool_path, tgt_lbl) in enumerate(pair_data):
            final_outdir = os.path.join(exp_outdir, cls_fool_path)
            os.makedirs(final_outdir, exist_ok=args.overwrite)
            data_path = os.path.join(projection_path, cls_fool_path)
            projections = np.load(os.path.join(data_path, "projections.npz"))["projections"] # steps x batch x 3 x 128 x 128
            projections = np.transpose(projections, axes=[0, 1, 3, 4, 2])

            confs = np.load(os.path.join(data_path, "statistics.npz"))["image_confs"] # steps x batch x # classes
            #prev_confs = confs[0, :, tgt_lbl]

            superpixel_labels = np.zeros(list(projections.shape[1:4]))
            for s_i, s_img in enumerate(projections[0]):
                superpixel_labels[s_i] = superpixel(s_img, pixels=400)
                n_lbls = int(superpixel_labels[s_i].max())+1
                used_regions = []
                conf_diffs = []
                out = C(F(convert_to_input(np.copy(projections[-1][s_i]))))
                end_conf = sm(out)[0][tgt_lbl].item()
                img = np.copy(projections[0][s_i])
                out = C(F(convert_to_input(img)))
                prev_conf = sm(out)[0][tgt_lbl].item() 
                save_img(img, "start.png")
                flipped = False
                while not flipped:
                    max_region = -1
                    max_conf = 0.0
                    for region in range(n_lbls):
                        if region in used_regions: continue
                        input_img = np.copy(img)
                        mask = superpixel_labels[s_i].astype(np.uint8) == region
                        input_img[mask] = projections[-1][s_i][mask]
                        out = C(F(convert_to_input(input_img)))
                        conf = sm(out)[0][tgt_lbl].item()
                        if conf > max_conf:
                            max_conf = conf
                            max_region = region
                    mask = superpixel_labels[s_i].astype(np.uint8) == max_region
                    img[mask] = projections[-1][s_i][mask]
                    out = C(F(convert_to_input(img)))
                    conf = sm(out)[0][tgt_lbl].item()
                    flipped = torch.argmax(out[0]) == tgt_lbl
                    conf_diffs.append(conf - prev_conf)
                    prev_conf = conf
                    used_regions.append(max_region)
                    save_img(img, "end.png")

                
                adjusted_conf_diffs = []
                for j, (diff, region) in enumerate(zip(conf_diffs, used_regions)):
                    test_img = np.copy(projections[0][s_i])
                    mask = superpixel_labels[s_i].astype(np.uint8) == region
                    test_img[mask] = projections[-1][s_i][mask]
                    out = C(F(convert_to_input(test_img)))
                    conf = sm(out)[0][tgt_lbl].item()
                    adjusted_conf = conf + ((diff-conf) / (j+1))
                    for k in range(j):
                        adjusted_conf_diffs[k] += ((diff-conf) / (j+1))
                    adjusted_conf_diffs.append(adjusted_conf)

                print(conf_diffs)
                print(adjusted_conf_diffs)
                conf_diff = adjusted_conf_diffs
                conf_diffs = np.array(conf_diffs)
                #conf_diffs -= conf_diffs.min()
                conf_diffs /= conf_diffs.max()
                out_img = np.copy(projections[0][s_i])
                highlight_img = np.zeros_like(projections[0][s_i])
                highlight_img[:, :, 1] = 1.0
                for j, (blend, region) in enumerate(zip(conf_diffs, used_regions)):
                    mask = superpixel_labels[s_i].astype(np.uint8) == region
                    out_img[mask] = out_img[mask] * (1-blend) + highlight_img[mask] * blend
                save_img(out_img, f"highlight_{s_i}.png")
            #exit()

            print(f"Exp {exp_i+1}/{len(experiments)} {cls_fool_i+1}/{len(pair_data)}: {cls_fool_path}")
        exp_time = f'{(perf_counter()-exp_start_time):.1f} s'
        print(f"Exp {exp_i+1}/{len(experiments)} run time: {exp_time}")
    all_time = f'{(perf_counter()-all_start_time):.1f} s'
    print(f"Total time to run: {all_time}")