import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
from time import perf_counter
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from models import Encoder
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from project import project

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--network', type=str, help='Network pickle filename', default="stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl")
    parser.add_argument('--backbone', type=str, default='../saved_models/vgg_backbone.pt')
    parser.add_argument('--classifier', type=str, default='../saved_models/vgg_classifier.pt')
    parser.add_argument('--dataset_root_train', type=str, default="../datasets/train/")
    parser.add_argument('--dataset_root_test', type=str, default="../datasets/test/")
    parser.add_argument('--subA', type=str, default='aglaope') # petiverana
    parser.add_argument('--subB', type=str, default='emma') # rosina
    parser.add_argument('--outdir', type=str, default='ganspace_output_optimization_avg_w')

    args = parser.parse_args()
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def load_images(paths):
    images = []
    for path in paths:
        img = transforms.Resize((128, 128))(Image.open(path)),
        images.append(np.array(img[0]))

    return np.array(images)

def load_data(dir_path, subspecies, type="train"):
    paths = load_json(os.path.join(dir_path, f"{type}_paths.json"))
    projections = np.transpose(np.load(os.path.join(dir_path, f"{type}_projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
    projections = (projections * 255).astype(np.uint8)
    ws = np.load(os.path.join(dir_path, f"{type}_ws.npz"))["ws"]
    perceptual_losses = np.load(os.path.join(dir_path, f"{type}_losses.npz"))["perceptual"]

    paths_filtered = []
    projections_filtered = []
    ws_filtered = []
    filtered_perceptual_losses = []
    for path, proj, w, loss in zip(paths, projections, ws, perceptual_losses):
        if path.split(os.path.sep)[-2] != subspecies: continue
        paths_filtered.append(path)
        projections_filtered.append(proj)
        ws_filtered.append(w)
        filtered_perceptual_losses.append(loss)

    originals = load_images(paths_filtered)
    cmbnd = zip(originals, projections_filtered, ws_filtered, filtered_perceptual_losses)
    all_sorted = sorted(cmbnd, key=lambda x: x[3])
    originals, projections_filtered, ws_filtered, _ = zip(*all_sorted)

    return originals, np.array(projections_filtered), np.array(ws_filtered)

def get_label_map(data_path):
    paths = []
    labels = []
    subspecies = []
    unique_subspecies = set()
    tmp_i = 0
    for root, dirs, files in os.walk(data_path):
        for f in files:
            parts = f.split("_")
            sub = root.split(os.path.sep)[-1]
            unique_subspecies.add(sub)
            subspecies.append(sub)
            path = os.path.join(root, f)
            paths.append(path)
    unique_subspecies = sorted(list(unique_subspecies))
    subspecies_to_lbl = {}
    for i, sub in enumerate(unique_subspecies):
        subspecies_to_lbl[sub] = i

    return subspecies_to_lbl

def get_pinciple_components(G):
    SAMPLES = 10000
    print(f'Computing {SAMPLES} W samples...')
    z_samples = np.random.RandomState(123).randn(SAMPLES, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    w_samples = w_samples[:, 0, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=False)      # [1, 1, C]

    w_center = w_samples - w_avg
    w_covariance = (1/SAMPLES)*(w_center.T @ w_center)

    eig_val, eig_vec = np.linalg.eig(w_covariance)
    
    return eig_vec, eig_val, w_avg

def create_images(G, w):
    w_input = torch.from_numpy(np.tile(w, (1, G.num_ws, 1))).cuda()
    synth_images = G.synthesis(w_input, noise_mode='const')
    synth_images = (synth_images + 1) * (1/2)
    synth_images = synth_images.clamp(0, 1)
    return synth_images
    #out_image = (synth_image * 255).detach().cpu().numpy().astype(np.uint8)
    
    #return np.transpose(out_image, axes=(1, 2, 0))

def classifier_preprocess(img):
    NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    return NORMALIZE(img)

def save_img(org, proj, synth):
    out_image = (synth * 255).detach().cpu().numpy().astype(np.uint8)
    out_image = np.transpose(out_image, axes=(1, 2, 0))

    final = np.concatenate((org, proj, out_image), axis=1)
    Image.fromarray(final).save("pc_img.png")

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    # Load Data
    sub_to_lbl_map = get_label_map(args.dataset_root_test)
    originals, projections, ws = load_data(args.outdir, args.subA, type="train")

    # Load Generator
    G, _, F, C = load_models(args.network, f_path=args.backbone, c_path=args.classifier)
    G = G.cuda()
    F = F.cuda()
    C = C.cuda()

    pcs, _, w_mean = get_pinciple_components(G)


    img_idx = 1
    w_add = np.zeros_like(pcs[0])
    pcs_to_save = []
    sm = nn.Softmax()
    synth = create_images(G, ws[img_idx])
    conf = sm(C(F(classifier_preprocess(synth)))[0])

    source_conf = conf[sub_to_lbl_map[args.subA]].item()
    target_conf = conf[sub_to_lbl_map[args.subB]].item()
    print(args.subA, source_conf)
    print(args.subB, target_conf)
    for pc_i, pc in enumerate(pcs):
        best_conf = target_conf
        s_conf = source_conf
        best_a = 0
        for w_pos in range(-20, 21, 1):
            a = w_pos / 10
            synth = create_images(G, ws[img_idx] + w_add + pc*a)
            conf = sm(C(F(classifier_preprocess(synth)))[0])

            cur_target_conf = conf[sub_to_lbl_map[args.subB]].item()
            if cur_target_conf > best_conf:
                best_conf = cur_target_conf
                s_conf = conf[sub_to_lbl_map[args.subA]].item()
                best_a = a

        if best_conf > target_conf:
            target_conf = best_conf
            print(f"Using PC {pc_i}")
            print(args.subA, s_conf)
            print(args.subB, best_conf)
            w_add += pc*best_a
            pcs_to_save.append(pc)
            save_img(originals[img_idx], projections[img_idx], synth[0])
        else:
            print(f'PC {pc_i} skipped')

    print(f'{len(pcs_to_save)} were used')
        


    

        
   

    




    