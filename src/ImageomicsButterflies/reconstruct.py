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
    parser.add_argument('--network', type=str, help='Network pickle filename', default="stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl")
    parser.add_argument('--encoder', help='encoder weights', type=str, default='encoder4editing/butterfly_training_2/checkpoints/best_model.pt')
    parser.add_argument('--encoder_type', help='backbone', type=str, default='e4e', choices=['e4e', 'simple'])
    parser.add_argument('--backbone', help='backbone', type=str, default='../saved_models/vgg_backbone.pt')
    parser.add_argument('--img', type=str, default="")
    parser.add_argument('--steps', type=int, default=500)

    args = parser.parse_args()
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

def initialize_w(G, path, E, F, num_samples=100, E_type='e4e'):
    if E is not None:
        if E_type == 'e4e':
            target_pil = Image.open(path).convert('RGB') # (res, res, # channels)
            images = encoder_transform()(target_pil)
            start_ws = E(images.unsqueeze(0).cuda())
            start_ws = start_ws.view(1, G.num_ws, -1).mean(1)
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
                loss = calc_percep_loss(F, NORMALIZE(synth_image), test_image_transform()(img).unsqueeze(0))
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

def reconstruct(path, lbl, E, G, F, steps=5, E_type='e4e'):
    start_ws = initialize_w(G, path, E, F, E_type=E_type)
    in_images = load_imgs(path, view="D")
    w_out, z_out, all_synth_images, pixel_losses, perceptual_losses, image_confs, _ = project(
        in_images,
        G,
        None,
        None,
        None,
        lbl,
        learn_param                = 'w',
        start_zs                   = None,
        start_ws                   = start_ws,
        num_steps                  = steps,
        init_lr                    = 0.001,
        img_to_img                 = True,
        batch                      = True,
        verbose                    = True,
        use_default_feat_extractor = True,
        multi_w                    = False #E_type == 'e4e'
    )
    
    return w_out[-1], all_synth_images[-1], pixel_losses[-1], perceptual_losses[-1]

def load_data(dset_path):
    paths = []
    labels = []
    subspecies = []
    unique_subspecies = set()
    tmp_i = 0
    for root, dirs, files in os.walk(dset_path):
        for f in files:
            parts = f.split("_")
            if parts[1] != 'D': continue
            sub = parts[2]
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

    return paths, labels

def load_encoder(path):
    E = load_e4e_standalone(path)
    return E

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)


    # Load autoencoder
    if args.encoder is not None:
        E = load_encoder(args.encoder)
    else:
        E = None

    # Load Generator
    G, _, F, _ = load_models(args.network, f_path=args.backbone, c_path=None)
    G = G.cuda()

    # Reconstruct
    w, synth_img, pix_loss, percep_loss = reconstruct(args.img, 0, E, G, F, steps=args.steps, E_type=args.encoder_type)
    
    Image.fromarray(np.transpose((synth_img.numpy() * 255).astype(np.uint8), axes=[2, 0, 1])).save("reconstruction.png")



    




    