import os
import torch
import torch.nn as nn
from copy import deepcopy, copy
from argparse import ArgumentParser
from time import perf_counter
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from models import Encoder
from encoder4editing.utils.model_utils import load_e4e_standalone
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from project import project

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def image_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def encoder_transform():
    return transforms.Compose([
	    transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--mode', type=str, default='filtered', choices=['filtered', 'original', 'original_nohybrid'])
    parser.add_argument('--hybrid', action="store_true", default=False)

    args = parser.parse_args()

    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
    elif args.mode == 'original':
        args.encoder = 'encoder4editing/butterfly_org_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original/00001-stylegan3-r-train_original_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.dataset_root_train = '../datasets/original/train/'
        args.dataset_root_test = '../datasets/original/test/'
        args.outdir = '../output/ganspace_reconstruction_original'
        args.backbone = None
        args.classifier = None
    elif args.mode == 'original_nohybrid':
        args.encoder = 'encoder4editing/butterfly_org_no_hybrid_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original_nohybrid/00000-stylegan3-r-train_original_no_hybrid_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = None
        args.classifier = None
        args.dataset_root_train = '../datasets/original_nohybrid/train/'
        args.dataset_root_test = '../datasets/original_nohybrid/test/'
        args.outdir = '../output/ganspace_reconstruction_original_nohybrid'
    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def load_images(paths):
    images = []
    for path in paths:
        img = transforms.Resize((128, 128))(Image.open(path)),
        images.append(np.array(img[0]))

    return np.array(images)

def load_data(dir_path, type="train"):
    paths = load_json(os.path.join(dir_path, f"{type}_paths.json"))
    projections = np.transpose(np.load(os.path.join(dir_path, f"{type}_projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
    projections = (projections * 255).astype(np.uint8)
    pixel_losses = np.load(os.path.join(dir_path, f"{type}_losses.npz"))["pixel"]
    perceptual_losses = np.load(os.path.join(dir_path, f"{type}_losses.npz"))["perceptual"]

    originals = load_images(paths)
    subspecies = list(map(lambda x: x.split(os.path.sep)[-2], paths))

    return subspecies, originals, projections, paths

def sort_data(subspecies, originals, projections, paths):
    cmbnd = zip(subspecies, originals, projections, paths)
    all_sorted = sorted(cmbnd, key=lambda x: x[0])
    return all_sorted

def analyze(subspecies, pixel_losses, perceptual_losses, outdir, type="train", thresh=9.5e-8, training_numbers=None):
    subspecies_successes = {}
    subspecies_totals = {}
    for sub, pix, percep in zip(subspecies, pixel_losses, perceptual_losses):
        if sub not in subspecies_successes:
            subspecies_successes[sub] = 0
            subspecies_totals[sub] = 0
        subspecies_totals[sub] += 1
        if percep <= thresh:
            subspecies_successes[sub] += 1
    labels = []
    success_percent = []
    for sub in subspecies_successes.keys():
        labels.append(sub)
        success_percent.append(subspecies_successes[sub] / subspecies_totals[sub])
        #print(sub, subspecies_totals[sub])

    sorted_data = zip(*sorted(zip(labels, success_percent, subspecies_totals.values(), subspecies_successes.values()), key=lambda x: x[2], reverse=True))
    labels, success_percent, totals, success_counts = list(sorted_data)
    if training_numbers is None:
        training_numbers = totals
    else:
        new_training_numbers = []
        for sub in labels:
            new_training_numbers.append(training_numbers[sub])
        training_numbers = new_training_numbers

    
    fig = plt.figure(figsize=(16, 9))

    plt.bar(labels, success_percent)
    ax = fig.gca()
    ax.set_xticklabels(labels=labels, rotation=60)
    ax.set_ylabel(f'{type} Reconstruction Success Rate')
    ax.set_ylim([0, 1.0])
    if type != "hybrids":
        ax2 = ax.twinx()
        ax2.plot(labels, training_numbers, c='red')
        ax2.set_ylabel('# Training Data')
    plt.savefig(os.path.join(outdir, "reconstruction_success_{:e}_{}.png".format(thresh, type)))
    plt.close()

    return subspecies_totals

def encode_and_gen(images, E, G):
    gen_imgs = []
    for img in images:
        images = encoder_transform()(Image.fromarray(img))
        start_ws = E(images.unsqueeze(0).cuda())
        start_ws = start_ws.view(1, G.num_ws, -1)[:, 0, :]
        gen_input = start_ws.unsqueeze(1).clone().repeat([1, G.mapping.num_ws, 1]).cuda()
        synth_images = G.synthesis(gen_input, noise_mode='const')
        synth_images = (synth_images + 1) * (1/2)
        synth_images = synth_images.clamp(0, 1)
        gen_img = (synth_images[0] * 255).cpu().detach().numpy().astype(np.uint8)
        gen_imgs.append(np.transpose(gen_img, axes=[1, 2, 0]))
    return gen_imgs

def parent_guess(F, C, org_imgs, encode_imgs, opt_imgs):
    top_5_data = []
    sm = nn.Softmax(dim=1).cuda()
    for org, enc, opt in zip(org_imgs, encode_imgs, opt_imgs):
        data = []
        out = C(F(image_transform()(Image.fromarray(org)).unsqueeze(0).cuda()))
        probs = sm(out)
        vals, idx = torch.topk(probs[0], 5)
        data.append([vals.detach().cpu().numpy(), idx.detach().cpu().numpy()])
        out = C(F(image_transform()(Image.fromarray(enc)).unsqueeze(0).cuda()))
        probs = sm(out)
        vals, idx = torch.topk(probs[0], 5)
        data.append([vals.detach().cpu().numpy(), idx.detach().cpu().numpy()])
        out = C(F(image_transform()(Image.fromarray(opt)).unsqueeze(0).cuda()))
        probs = sm(out)
        vals, idx = torch.topk(probs[0], 5)
        data.append([vals.detach().cpu().numpy(), idx.detach().cpu().numpy()])
        top_5_data.append(data)

    return top_5_data

def get_map(dset_root):
    subspecies = set()
    for root, dirs, files in os.walk(dset_root):
        for f in files:
            subspecies.add(root.split(os.path.sep)[-1])
    
    subspecies = sorted(list(subspecies))
    idx_to_sub_map = {}
    for i, sub in enumerate(subspecies):
        idx_to_sub_map[i] = sub

    return idx_to_sub_map

def create_text_row(width, height, text, padding):
        sub_text_img = (np.ones((height, width, 3)) * 255).astype(np.uint8)
        sub_text_img = Image.fromarray(sub_text_img)
        sub_text_img_dr = ImageDraw.Draw(sub_text_img)
        sub_text_img_dr.text((padding, padding), text, fill=(0, 0, 0))
        sub_text_img = np.array(sub_text_img)
        return sub_text_img

def create_col_img(img, data, header_text, idx_to_sub_map):
    TOP_ROW_HEIGHT = 30
    TEXT_ROW_HEIGHT = 20
    PAD = 2
    idx = data[1]
    vals = data[0]
    rows = []
    for i in range(len(idx)):
        txt = f"{idx_to_sub_map[idx[i]]}: {round(vals[i]*100, 2)}%"
        rows.append(txt)
    
    header_img = create_text_row(img.shape[1], TOP_ROW_HEIGHT, header_text, PAD)

    final_img = np.concatenate((header_img, img), axis=0)
    for row in rows:
        tmp = create_text_row(img.shape[1], TEXT_ROW_HEIGHT, row, PAD)
        final_img = np.concatenate((final_img, tmp), axis=0)

    return final_img

def save(top_5_data, paths, originals, encoded, optimized, idx_to_sub_map, outdir):
    final_img = None
    for path, data, org, enc, opt in zip(paths, top_5_data, originals, encoded, optimized):
        sub = path.split(os.path.sep)[-2]
        id = path.split(os.path.sep)[-1].split("_")[0]
        # Original
        org_img_col = create_col_img(org, data[0], "Original", idx_to_sub_map)
        enc_img_col = create_col_img(enc, data[1], "Encoder Only", idx_to_sub_map)
        opt_img_col = create_col_img(opt, data[2], "Enc + Opt", idx_to_sub_map)

        row_img = np.concatenate((org_img_col, enc_img_col, opt_img_col), axis=1)
        sub_text = create_text_row(row_img.shape[1], 30, sub, 2)
        row_img = np.concatenate((sub_text, row_img), axis=0)
        # Add border
        row_img[0, :] = np.array([255, 50, 50])
        row_img[-1, :] = np.array([255, 50, 50])
        row_img[:, 0] = np.array([255, 50, 50])
        row_img[:, -1] = np.array([255, 50, 50])
        Image.fromarray(row_img).save("test.png")
        
        if final_img is None:
            final_img = deepcopy(row_img)
        else:
            final_img = np.concatenate((final_img, row_img), axis=0)

    Image.fromarray(final_img).save(os.path.join(outdir, "hybrid_parent_discovery.png"))
        
        

            

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    # Load Data
    subspecies, originals, projections, paths = load_data(args.outdir, type="hybrids")

    # Sort Data
    sorted_data = sort_data(subspecies, originals, projections, paths)

    # idx to sub map
    idx_to_sub_map = get_map(args.dataset_root_train)

    # Load models
    G, _, F, C = load_models(args.network, f_path=args.backbone, c_path=args.classifier)
    E = load_e4e_standalone(args.encoder)
    G = G.cuda()
    E = E.cuda()

    # Get encoder ws
    gen_imgs = encode_and_gen(originals, E, G)

    G = None
    E = None
    C = C.cuda()
    F = F.cuda()

    top_5_data = parent_guess(F, C, originals, gen_imgs, projections)

    save(top_5_data, paths, originals, gen_imgs, projections, idx_to_sub_map, args.outdir)

        
   

    




    