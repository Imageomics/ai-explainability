import os
import torch
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
    parser.add_argument('--mode', type=str, default='filtered', choices=['filtered', 'original', 'original_nohybrid'])
    parser.add_argument('--hybrid', action="store_true", default=False)
    parser.add_argument('--sub', type=str, default=None)

    args = parser.parse_args()

    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
    elif args.mode == 'original':
        args.encoder = 'encoder4editing/butterfly_org_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original/00004-stylegan3-r-train_original_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.dataset_root_train = '../datasets/original/train/'
        args.dataset_root_test = '../datasets/original/test/'
        args.outdir = '../output/ganspace_reconstruction_original'
    elif args.mode == 'original_nohybrid':
        args.encoder = 'encoder4editing/butterfly_org_no_hybrid_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original_nohybrid/00006-stylegan3-r-train_original_no_hybrid_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
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

def load_data(dir_path, type="train", filter_sub=None):
    sub_txt = ""
    if filter_sub:
        sub_txt = f"_{filter_sub.split('_')[0]}"

        paths = load_json(os.path.join(dir_path, f"{type}{sub_txt}_paths.json"))
        projections = np.transpose(np.load(os.path.join(dir_path, f"{type}{sub_txt}_projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
        projections = (projections * 255).astype(np.uint8)
        pixel_losses = np.load(os.path.join(dir_path, f"{type}{sub_txt}_losses.npz"))["pixel"]
        perceptual_losses = np.load(os.path.join(dir_path, f"{type}{sub_txt}_losses.npz"))["perceptual"]

        sub_txt = f"_{filter_sub.split('_')[1]}"

        paths.extend(load_json(os.path.join(dir_path, f"{type}{sub_txt}_paths.json")))
        projections2 = np.transpose(np.load(os.path.join(dir_path, f"{type}{sub_txt}_projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
        projections2 = (projections2 * 255).astype(np.uint8)
        projections = np.concatenate((projections, projections2), axis=0)
        pixel_losses2 = np.load(os.path.join(dir_path, f"{type}{sub_txt}_losses.npz"))["pixel"]
        pixel_losses = np.concatenate((pixel_losses, pixel_losses2), axis=0)
        perceptual_losses2 = np.load(os.path.join(dir_path, f"{type}{sub_txt}_losses.npz"))["perceptual"]
        perceptual_losses = np.concatenate((perceptual_losses, perceptual_losses2), axis=0)

        originals = load_images(paths)
        subspecies = list(map(lambda x: x.split(os.path.sep)[-2], paths))
    else:
        paths = load_json(os.path.join(dir_path, f"{type}_paths.json"))
        projections = np.transpose(np.load(os.path.join(dir_path, f"{type}_projections.npz"))["projections"], axes=(1, 0, 3, 4, 2))[0]
        projections = (projections * 255).astype(np.uint8)
        pixel_losses = np.load(os.path.join(dir_path, f"{type}_losses.npz"))["pixel"]
        perceptual_losses = np.load(os.path.join(dir_path, f"{type}_losses.npz"))["perceptual"]

        originals = load_images(paths)
        subspecies = list(map(lambda x: x.split(os.path.sep)[-2], paths))

    return subspecies, originals, projections, pixel_losses, perceptual_losses

def sort_data(subspecies, originals, projections, pixel_losses, perceptual_losses):
    cmbnd = zip(subspecies, originals, projections, pixel_losses, perceptual_losses)
    all_sorted = sorted(cmbnd, key=lambda x: (x[0], x[4]))
    return all_sorted

def analyze(subspecies, pixel_losses, perceptual_losses, outdir, type="train", thresh=9.5e-8, training_numbers=None, filter_sub=""):
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
    plt.savefig(os.path.join(outdir, "reconstruction_success_{:e}_{}{}.png".format(thresh, type, filter_sub)))
    plt.close()

    return subspecies_totals

def perform_analysis(outdir, type="train", training_numbers=None, filter_sub=None):
    # Load Data
    subspecies, originals, projections, pixel_losses, perceptual_losses = load_data(outdir, type=type, filter_sub=filter_sub)

    # Analysis
    subspecies_totals = analyze(subspecies, pixel_losses, perceptual_losses, outdir, type, training_numbers=None, filter_sub=args.sub)

    # Sort Data
    sorted_data = sort_data(subspecies, originals, projections, pixel_losses, perceptual_losses)

    final_img = None
    for sub, org, proj, pix_loss, percep_loss in sorted_data:
        img_size = org.shape[:2]
        sub_text_img = (np.ones((img_size[0], 100, 3)) * 255).astype(np.uint8)
        sub_text_img = Image.fromarray(sub_text_img)
        sub_text_img_dr = ImageDraw.Draw(sub_text_img)
        sub_text_img_dr.text((4, (img_size[0] // 2)-10), sub, fill=(0, 0, 0))
        sub_text_img = np.array(sub_text_img)

        loss_text_img = (np.ones((img_size[0], 100, 3)) * 255).astype(np.uint8)
        loss_text_img = Image.fromarray(loss_text_img)
        loss_text_img_dr = ImageDraw.Draw(loss_text_img)
        loss_text_img_dr.text((4, (img_size[0] // 2)-10), "{:e}".format(percep_loss), fill=(0, 0, 0))
        loss_text_img_dr.text((4, (img_size[0] // 2)+10), str(round(pix_loss, 4)), fill=(0, 0, 0))
        loss_text_img = np.array(loss_text_img)

        row_img = np.concatenate((sub_text_img, org, proj, loss_text_img), axis=1)
        if final_img is None:
            final_img = row_img
        else:
            final_img = np.concatenate((final_img, row_img), axis=0)

    if filter_sub:
        Image.fromarray(final_img).save(os.path.join(outdir, f'reconstruction_visual_{type}_{filter_sub}.png'))
    else:
        Image.fromarray(final_img).save(os.path.join(outdir, f'reconstruction_visual_{type}.png'))
    return subspecies_totals

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    if args.hybrid:
        subspecies_totals = perform_analysis(args.outdir, type="hybrids", training_numbers=None)
    else:
        subspecies_totals = perform_analysis(args.outdir, type="train", training_numbers=None, filter_sub=args.sub)
        _ = perform_analysis(args.outdir, type="test", training_numbers=subspecies_totals, filter_sub=args.sub)

    

        
   

    




    