import os
import json

import torch
import PIL

import torchvision.transforms as transforms
import numpy as np

import stylegan3.dnnlib as dnnlib
import stylegan3.legacy as legacy

from models import Classifier, VGG16

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_latents(G, images, projected_ws, projected_zs, w_avg_samples=10000, use_rand_latents=False):
    if projected_zs is not None:
        projected_ws = G.mapping(projected_zs, None)
    elif projected_ws is None:
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        projected_zs = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        projected_zs = torch.from_numpy(projected_zs).cuda()
        w_samples = G.mapping(projected_zs, None)  # [N, L, C]
        w_samples = w_samples[:, :, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        projected_ws = torch.from_numpy(np.tile(np.mean(w_samples, axis=0, keepdims=True)[0], (len(images), 1, 1))).cuda()      # [1, 1, C]
        projected_zs = projected_zs.mean(0, keepdim=True).repeat([len(images), 1])

    return projected_zs, projected_ws

#----------------------------------------------------------------------------
def load_img(img_path, resolution=128):
    target_pil = PIL.Image.open(img_path).convert('RGB') # (res, res, # channels)
    target_pil = target_pil.resize((resolution, resolution), PIL.Image.LANCZOS)
    return transforms.ToTensor()(target_pil)
#----------------------------------------------------------------------------
def load_imgs(img_dir, view="D", max_size=64, resolution=128):
    if img_dir is None:
        return None

    if os.path.isfile(img_dir):
        img = load_img(img_dir, resolution)
        return img.unsqueeze(0).cuda()
    
    batch = None
    i = 0
    for root, dirs, paths in os.walk(img_dir):
        for path in paths:
            fview = path.split(".")[0].split("_")[1]
            if fview != view: continue
            i += 1

            full_path = os.path.join(root, path)
            img = load_img(full_path, resolution)
            if batch is None:
                batch = img.unsqueeze(0)
            else:
                batch = torch.cat((batch, img.unsqueeze(0)), axis=0)

            if i >= max_size: break
            
    batch = batch.cuda()
    return batch
#----------------------------------------------------------------------------

def load_latents(G, latent_path=None, avg_samples=10000, batch_size=1):
    projected_ws = None
    projected_zs = None
    if latent_path is None:
        print(f'Computing W midpoint and stddev using {avg_samples} samples...')
        projected_zs = np.random.RandomState(123).randn(avg_samples, G.z_dim)
        projected_zs = torch.from_numpy(projected_zs).cuda()
        projected_zs = projected_zs.mean(0, keepdim=True).repeat([batch_size, 1])
        projected_ws = G.mapping(projected_zs, None)[:, 0, :]  # [N, L, C]
        assert projected_zs.shape == (batch_size, G.z_dim), "Z projection shape incorrect"
        assert projected_zs.shape == (batch_size, G.w_dim), "W projection shape incorrect"
        
        return projected_zs, projected_ws

    latents = np.load(latent_path)
    if 'z' in latents.keys():
        projected_zs = torch.tensor(latents['z']).cuda()
    if 'w' in latents.keys():
        projected_ws = torch.tensor(latents['w']).cuda()

    if projected_zs is not None:
        projected_ws = G.mapping(projected_zs, None)[:, 0, :]
        
    return projected_zs, projected_ws

def load_models(gen_path=None, f_path=None, c_path=None, num_classes=27):
    # Load networks.
    G = None
    D = None
    if gen_path is not None:
        print('Loading networks from "%s"...' % gen_path)
        with dnnlib.util.open_url(gen_path) as fp:
            net = legacy.load_network_pkl(fp)
            # Generator
            G = net['G_ema'].requires_grad_(False).cuda() # type: ignore
            G.eval()

            # Discriminator
            D = net['D'].requires_grad_(False).cuda() # type: ignore
            D.eval()

    # Feature Extractor
    F = None
    if f_path is not None:
        f_weights = torch.load(f_path)
        F = VGG16(pretrain=False).cuda()
        F.load_state_dict(f_weights)
        F.eval()

    # Classifier
    C = None
    if c_path is not None:
        c_weights = torch.load(c_path)
        C = Classifier(F.in_features, num_classes).cuda()
        C.load_state_dict(c_weights)
        C.eval()

    return G, D, F, C