import os
from argparse import ArgumentParser
from time import perf_counter
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
import imageio

from models import Encoder
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from data_tools import NORMALIZE
from project import project

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--mode', type=str, default='filtered', choices=['afhqv2', 'filtered', 'filtered_cond', 'filtered_cond_v2', 'original', 'original_nohybrid'])
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--src_sub', type=str, default='aglaope')
    parser.add_argument('--tgt_sub', type=str, default='emma')
    parser.add_argument('--filter_sub', action='store_true', default=False)
    parser.add_argument('--save_freq', type=int, default=10)

    # Best lambdas
    """
    lr: 0.01
    class: 10.0
    image diff: 100
    image diff entropy: 10
    counter factual: 1.0
    smooth: 0.00001
    """

    args = parser.parse_args()

    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.backbone = '../saved_models/vgg_smooth_filtered_backbone.pt'
        args.classifier = '../saved_models/vgg_smooth_filtered_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
        args.num_classes = 27
    elif args.mode == 'filtered_cond':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w_cond/checkpoints/best_model.pt'
        args.network = 'stylegan3_cond/butterfly_training_runs/00029-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered_cond'
        args.num_classes = 27
    elif args.mode == 'filtered_cond_v2':
        args.encoder = None #'encoder4editing/butterfly_training_only_one_w_cond/checkpoints/best_model.pt'
        args.network = 'stylegan3_cond/butterfly_training_runs/00041-stylegan3-r-train_128_128-gpus2-batch64-gamma6.6/network-snapshot-002016.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered_cond_v2'
        args.num_classes = 27
    elif args.mode == 'original':
        args.encoder = 'encoder4editing/butterfly_org_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original/00004-stylegan3-r-train_original_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone_original.pt'
        args.classifier = '../saved_models/vgg_classifier_original.pt'
        args.dataset_root_train = '../datasets/original/train/'
        args.dataset_root_test = '../datasets/original/test/'
        args.outdir = '../output/ganspace_reconstruction_original'
        args.num_classes = 37
    elif args.mode == 'original_nohybrid':
        args.encoder = 'encoder4editing/butterfly_org_no_hybrid_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original_nohybrid/00006-stylegan3-r-train_original_no_hybrid_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone_original_nohybrid.pt'
        args.classifier = '../saved_models/vgg_classifier_original_nohybrid.pt'
        args.dataset_root_train = '../datasets/original_nohybrid/train/'
        args.dataset_root_test = '../datasets/original_nohybrid/test/'
        args.outdir = '../output/ganspace_reconstruction_original_nohybrid'
        args.num_classes = 34
    elif args.mode == 'afhqv2':
        args.encoder = '/home/carlyn.1/ImagenomicsButterflies/src/encoder4editing/afhqv2_training_only_one_w/checkpoints/best_model.pt'
        args.network = 'stylegan3_org/afhqv2_model/stylegan3-r-afhqv2-512x512.pkl'
        args.backbone = '/home/carlyn.1/ImagenomicsButterflies/output/afhqv2_classifier_4/vgg_afhqv2_backbone.pt'
        args.classifier = '/home/carlyn.1/ImagenomicsButterflies/output/afhqv2_classifier_4/vgg_afhqv2_classifier.pt'
        args.dataset_root_train = '../datasets/afhqv2/train/'
        args.dataset_root_test = '../datasets/afhqv2/test/'
        args.outdir = '../output/ganspace_reconstruction_afhqv2'
        args.num_classes = 3
        args.res = 512
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def get_sub_lbl_map(dset):
    subspecies = set()
    for root, dirs, files in os.walk(dset):
        for f in files:
            subspecies.add(root.split(os.path.sep)[-1])
    
    sub_lbl_map = {}
    subspecies = sorted(list(subspecies))
    for i, sub in enumerate(subspecies):
        sub_lbl_map[sub] = i

    return sub_lbl_map


def load_images(paths):
    images = []
    for path in paths:
        img = transforms.Resize((128, 128))(Image.open(path)),
        images.append(np.array(img[0]))

    return np.array(images)

def load_data(dir_path, type="train", filter_sub=None):
    add_sub = f"{filter_sub}_" if filter_sub is not None else ""
    paths = load_json(os.path.join(dir_path, f"{type}_{add_sub}paths.json"))
    projections = np.transpose(np.load(os.path.join(dir_path, f"{type}_{add_sub}projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
    projections = (projections * 255).astype(np.uint8)
    ws = np.load(os.path.join(dir_path, f"{type}_{add_sub}ws.npz"))["ws"]

    originals = load_images(paths)
    subspecies = list(map(lambda x: x.split(os.path.sep)[-2], paths))

    return subspecies, originals, projections, ws

def create_images(G, w, no_repeat=False):
    if no_repeat:
        w_input = w
    else:
        w_input = w.unsqueeze(1).repeat((1, G.num_ws, 1)).cuda()
    synth_images = G.synthesis(w_input, noise_mode='const')
    synth_images = (synth_images + 1) * (1/2)
    synth_images = synth_images.clamp(0, 1)
    return synth_images

def to_numpy_img(img):
    return np.transpose(img.detach().cpu().numpy()*255, axes=(1, 2, 0)).astype(np.uint8)

def save_image(img, path):
    np_img = to_numpy_img(img[0])
    Image.fromarray(np_img).save(path)

def ten_grayscale(x):
    return x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114

def grayscale(x):
    rv = np.zeros_like(x[:3])
    rv = x[:, :, 0] * 0.299 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.114
    return rv

def get_diff_img(a, b):
    diff_img = (grayscale(b) - grayscale(a)).astype(np.float)
    diff_pos = np.copy(diff_img)
    diff_pos[diff_pos < 0] = 0
    diff_neg = -np.copy(diff_img)
    diff_neg[diff_neg < 0] = 0
    
    THRESH = 0.25
    diff_pos -= diff_pos.min()
    diff_pos /= diff_pos.max()
    diff_pos[diff_pos < THRESH] = 0.0
    diff_pos *= 255

    diff_neg -= diff_neg.min()
    diff_neg /= diff_neg.max()
    diff_neg[diff_neg < THRESH] = 0.0
    diff_neg *= 255

    diff_img = np.concatenate((np.expand_dims(diff_neg, 2), np.expand_dims(np.zeros_like(diff_neg), 2), np.expand_dims(diff_pos, 2)), axis=2).astype(np.uint8)

    #diff_img = np.tile(np.expand_dims(diff_img, 2), (1, 1, 3)).astype(np.uint8)
    return diff_img


def pca(data):
    data = np.array(data)
    mu = data.mean(0)
    center = data - mu
    cov = (1/len(data))*(center.T @ center)
    eig_val, eig_vec = np.linalg.eig(cov)
    W = np.copy(eig_vec)

    return (data - mu) @ W.T, W, mu, eig_vec, eig_val

def add_text(img, text=""):
    TEXT_HEIGHT = 20
    PAD = 4
    img_size = img.shape[:2]
    text_img = (np.ones((TEXT_HEIGHT, img_size[1], 3)) * 255).astype(np.uint8)
    text_img = Image.fromarray(text_img)
    text_img_dr = ImageDraw.Draw(text_img)
    text_img_dr.text((PAD, PAD), text, fill=(0, 0, 0))
    text_img = np.array(text_img)
    return np.concatenate((text_img, img), axis=0)

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    """
    ?Question: Will a couterfactual in the pca ganspace be more distinguishable?
    
    Control: Show counterfactual in w space
    Experiment: Show counterfactual in pca w space

    !Answer: Seems that there is either no difference or pca is worse.
    ! In both cases, adversarial examples seem to take over
    """

    sub_lbl_map = get_sub_lbl_map(args.dataset_root_train)
    if args.filter_sub:
        subspecies, originals, projections, ws = load_data(args.outdir, type="train", filter_sub=args.src_sub)
        src_ws_list = list(ws)
        subspecies, originals, projections, ws = load_data(args.outdir, type="train", filter_sub=args.tgt_sub)
        tgt_ws_list = list(ws)
    else:
        subspecies, originals, projections, ws = load_data(args.outdir, type="train")
        src_ws_list = list(map(lambda x: x[1], filter(lambda x: x[0]==args.src_sub, zip(subspecies, ws))))
        tgt_ws_list = list(map(lambda x: x[1], filter(lambda x: x[0]==args.tgt_sub, zip(subspecies, ws))))
    src_w = np.array(src_ws_list)
    tgt_w = np.array(tgt_ws_list)
    
    G, _, F, C = load_models(args.network, args.backbone, args.classifier, args.num_classes)

    tgt_lbl = torch.tensor([sub_lbl_map[args.tgt_sub]]).cuda()
    src_lbl = torch.tensor([sub_lbl_map[args.src_sub]]).cuda()

    attributes = []
    directions = []
    M = args.M
    shift_size = 5
    d_scale_min = np.min(tgt_w, axis=0)
    d_scale_max = np.max(tgt_w, axis=0)
    thresh = 5
    class_attributes = (G.w_dim // 2) // args.num_classes
    target_attributes = np.arange(class_attributes) + (args.num_classes * tgt_lbl.cpu().item())
    attributes.extend(target_attributes)
    directions.extend([0, 0])

    new_src_w = []
    while len(attributes) < M and len(src_w) > 0 and False:
        diffs = np.zeros((len(src_w), src_w.shape[-1], 2))
        for w_dim in range(src_w.shape[-1]):
            if w_dim in attributes: continue
            for i, w in enumerate(src_w):
                w_ten = torch.tensor(w).cuda()
                d_pos = w_ten.clone().detach()
                d = (d_scale_max[w_dim] - d_pos[w_dim]) * shift_size
                d_pos[w_dim] += torch.tensor(d).cuda()
                d_neg = w_ten.clone().detach()
                d = (d_scale_min[w_dim] - d_neg[w_dim]) * shift_size
                d_neg[w_dim] += torch.tensor(d).cuda()
                all_ws = torch.cat((w_ten.unsqueeze(0), (d_pos).unsqueeze(0), (d_neg).unsqueeze(0)), dim=0)
                imgs = create_images(G, all_ws)
                probs = C(F(NORMALIZE(imgs)))[:, tgt_lbl.item()]
                diffs[i, w_dim, 0] = (probs[1] - probs[0]).item()
                diffs[i, w_dim, 1] = (probs[2] - probs[0]).item()
        diffs_avg = diffs.mean(0)
        for w_dim in range(src_w.shape[-1]):
            if w_dim in attributes: continue
            if diffs_avg[w_dim, 0] > 0 and diffs_avg[w_dim, 1] > 0:
                diffs_avg[w_dim, :] = 0.0
        att, direct = np.unravel_index(np.argmax(diffs_avg, axis=None), diffs_avg.shape)
        attributes.append(att)
        directions.append(direct)
        #for i in reversed(range(len(src_w))):
        #    if diffs[i, att, direct] > thresh:
        #        np.delete(src_w, i)

        print(attributes)
        print(directions)
    
    sm = nn.Softmax(dim=1)
    video_imgs = []
    for shift_size in range(50):
        final_img = None
        for i, w in enumerate(src_w):
            w_ten = torch.tensor(w).cuda()
            img_org = create_images(G, w_ten.unsqueeze(0).cuda())
            org_prob = sm(C(F(NORMALIZE(img_org))))[0][tgt_lbl.item()].item()
            org_prob = round(org_prob * 100, 2)
            img_org_np = to_numpy_img(img_org[0])
            img_org_np_text = add_text(img_org_np, f"SRC: {org_prob}%")
            row_img = img_org_np_text
            all_atts = w_ten.clone().detach()
            for att, direct in zip(attributes, directions):
                d = w_ten.clone().detach()
                v = torch.tensor(d_scale_max[att]).cuda() if direct == 0 else torch.tensor(d_scale_min[att]).cuda()
                #d_delta = (v - d[att]) * shift_size
                d_delta = shift_size
                d[att] += d_delta
                w_ten = torch.tensor(w).cuda()
                all_atts[att] += d_delta
                img_new = create_images(G, torch.tensor(d).unsqueeze(0))
                new_prob = sm(C(F(NORMALIZE(img_new))))[0][tgt_lbl.item()].item()
                new_prob = round(new_prob * 100, 2)
                img_new_np = to_numpy_img(img_new[0])
                img_new_np_text = add_text(img_new_np, f"#{att}: {new_prob}%")
                row_img = np.concatenate((row_img, img_new_np_text), axis=1)
                diff_img = get_diff_img(img_org_np, img_new_np)
                diff_img_text = add_text(diff_img, f"#{att} Diff img")
                row_img = np.concatenate((row_img, diff_img_text), axis=1)
            all_atts_img = create_images(G, torch.tensor(all_atts).unsqueeze(0))
            all_atts_prob = sm(C(F(NORMALIZE(all_atts_img))))[0][tgt_lbl.item()].item()
            all_atts_prob = round(all_atts_prob * 100, 2)
            all_atts_np = to_numpy_img(all_atts_img[0])
            all_atts_text = add_text(all_atts_np, f"All: {all_atts_prob}%")
            row_img = np.concatenate((row_img, all_atts_text), axis=1)
            if final_img is None:
                final_img = row_img
            else:
                final_img = np.concatenate((final_img, row_img), axis=0)
        video_imgs.append(final_img)
    Image.fromarray(final_img).save("attributes.png")

    video = imageio.get_writer("attributes.mp4", mode='I', fps=5, codec='libx264')
    for frame in video_imgs:
        video.append_data(frame)
    video.close()
            