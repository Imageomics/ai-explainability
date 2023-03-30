import os
from argparse import ArgumentParser
from time import perf_counter
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models import Encoder
from encoder4editing.utils.model_utils import load_e4e_standalone
from helpers import set_random_seed, cuda_setup
from loading_helpers import save_json, load_json, load_imgs, load_latents, load_models
from data_tools import NORMALIZE, test_image_transform
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
    parser.add_argument('--mode', type=str, default='filtered', choices=['afhqv2', 'filtered', 'filtered_cond', 'filtered_cond_v2', 'original', 'original_nohybrid'])
    parser.add_argument('--sub', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--phase', type=str, default='both', choices=['both', 'train', 'test'])

    args = parser.parse_args()

    args.encoder_type = 'e4e'
    if args.mode == 'filtered':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w/checkpoints/iteration_200000.pt'
        args.network = 'stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
        args.dataset_root_train = '../datasets/train/'
        args.dataset_root_test = '../datasets/test/'
        args.outdir = '../output/ganspace_reconstruction_filtered'
    elif args.mode == 'filtered_cond':
        args.encoder = 'encoder4editing/butterfly_training_only_one_w_cond/checkpoints/best_model.pt'
        args.network = 'stylegan3_cond/butterfly_training_runs/00029-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl'
        args.backbone = '../saved_models/vgg_backbone.pt'
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

def sample_w_global_mean(samples, G):
    print(f'Computing {args.samples} W samples...')
    z_samples = np.random.RandomState(123).randn(args.samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    w_samples = w_samples[:, 0, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=False)
    return w_avg

def pca(data, dim=2):
    data = np.array(data)
    mu = data.mean(0)
    center = data - mu
    cov = (1/len(data))*(center.T @ center)
    eig_val, eig_vec = np.linalg.eig(cov)
    W = eig_vec[:dim]

    return data @ W.T

def pca_analysis(args, G):
    print(f'Computing {args.samples} W samples...')
    z_samples = np.random.RandomState(123).randn(args.samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).cuda(), None)  # [N, L, C]
    w_samples = w_samples[:, 0, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=False)      # [1, 1, C]

    w_center = w_samples - w_avg
    w_covariance = (1/args.samples)*(w_center.T @ w_center)

    eig_val, eig_vec = np.linalg.eig(w_covariance)
    eig_val_total = eig_val.sum()

    accum = 0
    for i, val in enumerate(eig_val):
        accum += val
        percent = accum / eig_val_total
        if percent >= 0.99:
            print(f"Top {i+1} eignvectors can recreate {round(percent, 4)*100}% of the data")
            break

def get_images(G, w):
    synth_images = G.synthesis(w, noise_mode='const')
    synth_images = (synth_images + 1) * (1/2)
    synth_images = synth_images.clamp(0, 1)
    return synth_images

def calc_percep_loss(F, img1, img2):
    feat1 = F(img1)
    feat2 = F(img2)
    MSELoss = nn.MSELoss().cuda()
    return MSELoss(feat1, feat2).item()

def initialize_w(G, path, E, F, num_samples=100, E_type='e4e', is_butterfly=True):
    IMG_SIZE = 256 if is_butterfly else 512
    if E is not None:
        if E_type == 'e4e':
            target_pil = Image.open(path).convert('RGB') # (res, res, # channels)
            images = encoder_transform()(target_pil)
            start_ws = E(images.unsqueeze(0).cuda())
            start_ws = start_ws.view(1, G.num_ws, -1)[:, 0, :]
        elif E_type == 'simple':
            pass
    else:
        images = load_imgs(path, view="D")
        projected_zs = np.random.RandomState(123).randn(num_samples, G.z_dim)
        projected_zs = torch.from_numpy(projected_zs).cuda()
        projected_zs = projected_zs.repeat([len(images), 1])
        projected_ws = G.mapping(projected_zs, None)[:, :1, :]  # [N, L, C]
        start_ws = []
        for i in range(len(images)):
            img = images[i]
            ws = projected_ws[:, i, :]
            best_w = None
            best_loss = 100000
            for w in ws:
                w_in = w.repeat([1, G.mapping.num_ws, 1])
                synth_image = get_images(G, w_in)
                loss = calc_percep_loss(F, NORMALIZE(synth_image), test_image_transform(resize_size=IMG_SIZE)(img).unsqueeze(0))
                if loss < best_loss:
                    best_loss = loss
                    best_w = w
            start_ws.append(w.detach().cpu().numpy())
        start_ws = torch.from_numpy(np.array(start_ws)).cuda()
    return start_ws
    #start_zs, start_ws = load_latents(G, None, batch_size=len(images))

def visualize(data, labels):
    for i in range(max(labels)+1):
        dps = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == i, zip(labels, latents)))))
        plt.scatter(dps[:, 0], dps[:, 1], label=i)
    plt.savefig("pca.png")

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

def load_encoder(path):
    E = load_e4e_standalone(path)
    return E

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    #is_butterfly = not (args.mode in ['afhqv2'])
    #paths, labels = load_data(args.dataset_root_train, sub_filter=args.sub, is_butterfly=is_butterfly)


    # Load autoencoder
    if args.encoder is not None:
        E = load_encoder(args.encoder)
    else:
        E = None

    # Load Generator
    G, _, F, _ = load_models(args.network, f_path=args.backbone, c_path=None)
    G = G.cuda()


    def do_reconstruction(paths, labels, args, save_lbl="train", verbose=False, limit=0, is_butterfly=True):
        results = []
        lbl_count = {}
        for path, lbl in tqdm(zip(paths, labels), desc=f"Reconstructing {save_lbl} Images", ncols=100):
            if limit > 0:
                if lbl not in lbl_count:
                    lbl_count[lbl] = 0
                if lbl_count[lbl] >= limit: continue
                lbl_count[lbl] += 1
            w, synth_img, pix_loss, percep_loss = reconstruct(path, lbl, E, G, F, steps=args.steps, E_type=args.encoder_type, verbose=verbose, is_butterfly=is_butterfly)
            w = w[0].numpy()
            results.append([path, w, synth_img, pix_loss, percep_loss])

        paths = list(map(lambda x: x[0], results))
        save_json(paths, f'{args.outdir}/{save_lbl}_paths.json')

        ws = np.array(list(map(lambda x: x[1], results)))
        np.savez(f'{args.outdir}/{save_lbl}_ws.npz', ws=ws)

        projections = np.array(list(map(lambda x: x[2], results)))
        np.savez(f'{args.outdir}/{save_lbl}_projections.npz', projections=projections)

        pixel_losses = np.array(list(map(lambda x: x[3], results)))
        perceptual_losses = np.array(list(map(lambda x: x[4], results)))
        np.savez(f'{args.outdir}/{save_lbl}_losses.npz', pixel=pixel_losses, perceptual=perceptual_losses)


    is_butterfly = not (args.mode in ['afhqv2'])
    if args.hybrid and args.sub is None:
        paths, labels = load_data("../datasets/hybrids")
        do_reconstruction(paths, labels, args, save_lbl="hybrids", verbose=args.verbose, is_butterfly=is_butterfly)

    # Load Data
    if args.phase in ['both', 'train']:
        paths, labels = load_data(args.dataset_root_train, sub_filter=args.sub, is_butterfly=is_butterfly)
        save_lbl = "train"
        if args.sub:
            save_lbl += f"_{args.sub}"
        do_reconstruction(paths, labels, args, save_lbl=save_lbl, verbose=args.verbose, limit=args.limit, is_butterfly=is_butterfly)

    # Load Data
    if args.phase in ['both', 'test']:
        paths, labels = load_data(args.dataset_root_test, sub_filter=args.sub, is_butterfly=is_butterfly)
        save_lbl = "test"
        if args.sub:
            save_lbl += f"_{args.sub}"
        do_reconstruction(paths, labels, args, save_lbl=save_lbl, verbose=args.verbose, limit=args.limit, is_butterfly=is_butterfly)



    




    