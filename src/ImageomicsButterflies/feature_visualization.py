import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from data_tools import NORMALIZE, UNNORMALIZE, test_image_transform
from project import project

def encoder_transform():
    return transforms.Compose([
	    transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--hybrid', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='filtered', choices=['afhqv2', 'filtered', 'filtered_no_pre', 'filtered_cond', 'filtered_cond_v2', 'original', 'original_nohybrid'])
    parser.add_argument('--sub', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--phase', type=str, default='both', choices=['both', 'train', 'test'])
    parser.add_argument('--img_path', type=str, default='../datasets/train/aglaope/10428242_D_lowres.png')

    args = parser.parse_args()

    args.encoder_type = 'e4e'
    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_smooth_filtered_backbone.pt'
        args.classifier = '../saved_models/vgg_smooth_filtered_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
    elif args.mode == 'filtered_no_pre':
        args.encoder = None
        args.network = None
        args.backbone = '../output/filtered_classifier_6/vgg_filtered_backbone.pt'
        args.classifier = '../output/filtered_classifier_6/vgg_filtered_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
    elif args.mode == 'filtered_cond':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w_cond/checkpoints/best_model.pt'
        args.network = 'stylegan3_cond/butterfly_training_runs/00029-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.classifier = '../saved_models/vgg_classifier.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered_cond'
    elif args.mode == 'filtered_cond_v2':
        args.encoder = None
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
        args.dataset_root_train = '../datasets/original/train/'
        args.dataset_root_test = '../datasets/original/test/'
        args.outdir = '../output/ganspace_reconstruction_original'
    elif args.mode == 'original_nohybrid':
        args.encoder = 'encoder4editing/butterfly_org_no_hybrid_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3_org/butterfly_training_runs_original_nohybrid/00006-stylegan3-r-train_original_no_hybrid_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone_original_nohybrid.pt'
        args.dataset_root_train = '../datasets/original_nohybrid/train/'
        args.dataset_root_test = '../datasets/original_nohybrid/test/'
        args.outdir = '../output/ganspace_reconstruction_original_nohybrid'
    elif args.mode == 'afhqv2':
        args.encoder = '/home/carlyn.1/ImagenomicsButterflies/src/encoder4editing/afhqv2_training_only_one_w/checkpoints/best_model.pt'
        args.network = 'stylegan3_org/afhqv2_model/stylegan3-r-afhqv2-512x512.pkl'
        args.backbone = '/home/carlyn.1/ImagenomicsButterflies/output/afhqv2_classifier_4/vgg_afhqv2_backbone.pt'
        args.dataset_root_train = '../datasets/afhqv2/train/'
        args.dataset_root_test = '../datasets/afhqv2/test/'
        args.outdir = '../output/ganspace_reconstruction_afhqv2'

    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def load_img(path):
    return transforms.Compose([
	    transforms.Resize((128, 128)),
	    transforms.ToTensor()])(Image.open(path))

def reconstruct(path, lbl, E, G, F, steps=5, E_type='e4e', verbose=False, is_butterfly=True):
    IMG_SIZE = 128 if is_butterfly else 512
    start_ws = initialize_w(G, path, E, F, E_type=E_type, is_butterfly=is_butterfly)
    in_images = load_imgs(path, view="D", resolution=IMG_SIZE)
    w_out, _, all_synth_images, pixel_losses, perceptual_losses, _, _ = project(
        in_images,
        G,
        None,
        F,
        None,
        lbl,
        learn_param                = 'w',
        start_zs                   = None,
        start_ws                   = start_ws,
        num_steps                  = steps,
        init_lr                    = 0.001,
        img_to_img                 = True,
        batch                      = True,
        verbose                    = verbose,
        use_default_feat_extractor = False,
        multi_w                    = False #E_type == 'e4e'
    )
    
    return w_out[-1], all_synth_images[-1], pixel_losses[-1], perceptual_losses[-1]

def load_data(dset_path, sub_filter=None, is_butterfly=True):
    paths = []
    labels = []
    subspecies = []
    unique_subspecies = set()
    tmp_i = 0
    for root, _, files in os.walk(dset_path):
        for f in files:
            parts = f.split("_")
            if parts[1] != 'D' and is_butterfly: continue
            sub = root.split(os.path.sep)[-1]
            unique_subspecies.add(sub)
            subspecies.append(sub)
            path = os.path.join(root, f)
            paths.append(path)
    unique_subspecies = sorted(list(unique_subspecies))
    subspecies_to_lbl = {}
    for i, sub in enumerate(unique_subspecies):
        subspecies_to_lbl[sub] = i

    for sub in subspecies:
        labels.append(subspecies_to_lbl[sub])

    if sub_filter:
        new_paths = []
        new_labels = []
        for path in paths:
            sub = path.split(os.path.sep)[-2]
            if sub != sub_filter: continue
            new_paths.append(path)
            new_labels.append(subspecies_to_lbl[sub])
        return new_paths, new_labels

    return paths, labels

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)


    # Load Model
    _, _, F, C = load_models(None, f_path=args.backbone, c_path=args.classifier)

    # Load Data
    img = load_img(args.img_path)
    img_vis = torch.rand_like(img).cuda().requires_grad_()
    print(img_vis.shape)
    optimizer = torch.optim.Adam([img_vis], betas=(0.9, 0.999), lr=args.lr)

    rotate = transforms.RandomRotation(90)
    jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    for i in range(args.steps):
        out = C(F(rotate(jitter(img_vis)).unsqueeze(0)))[0]
        loss = -out[0]
        
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            tmp = torch.clamp(UNNORMALIZE(img_vis), 0, 1)
            transforms.ToPILImage()(tmp).save("tmp.png")



    