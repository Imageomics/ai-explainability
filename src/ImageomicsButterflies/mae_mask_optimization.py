from distutils.archive_util import make_archive
import sys
import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from helpers import set_random_seed, cuda_setup
from loading_helpers import load_models
from data_tools import NORMALIZE, UNNORMALIZE, rgb_img_loader
sys.path.append("mae")
import models_mae

def mae_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NORMALIZE
    ])

def classifier_transform(crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def load_all_models(backbone, classifier, mae):
    _, _, F, C = load_models(None, backbone, classifier)
    mae_model = model = getattr(models_mae, 'mae_vit_base_patch16')()
    ckpt = torch.load(mae, map_location='cpu')
    msg = model.load_state_dict(ckpt['model'], strict=False)
    print(msg)
    return F, C, mae_model

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument('--backbone', help='backbone', type=str, default='../saved_models/vgg_backbone.pt')
    parser.add_argument('--classifier', help='classifier', type=str, default='../saved_models/vgg_classifier.pt')
    parser.add_argument('--mae', default='mae/output_dir/checkpoint-99.pth')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--img', type=str, default='../datasets/train/aglaope/10428242_D_lowres.png')
    parser.add_argument('--lbl', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--no_mae', action='store_true', default=False)
    parser.add_argument('--criteria', type=str, default='dist')
    args = parser.parse_args()

    args.gpu_ids = ",".join(map(lambda x: str(x), args.gpu_ids))
    return args

def load_input(path):
    img = rgb_img_loader(path)
    return mae_transform()(img).unsqueeze(0)



def reconstruct(input_img, mask_ids, ids_restore, mae, no_mae=False):
    x = mae.patch_embed(input_img)
    num_patches = x.shape[1]
    x = x + mae.pos_embed[:, 1:, :]

    x = torch.gather(x, dim=1, index=mask_ids.unsqueeze(-1).repeat(1, 1, x.shape[2]))
    

    cls_token = mae.cls_token + mae.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # apply Transformer blocks
    for blk in mae.blocks:
        x = blk(x)
    x = mae.norm(x)

    rv = mae.forward_decoder(x, ids_restore)  # [N, L, p*p*3]
    rv = mae.unpatchify(rv)

    mask = torch.ones((1, num_patches)).cuda()
    for mask_id in mask_ids[0]:
        mask[0][mask_id.item()] = 0
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = mae.unpatchify(mask)
    if no_mae:
        return input_img * (1-mask)
    rv = rv * mask + input_img * (1-mask)
    return rv

def to_pil(tensor_img):
    tensor_img = UNNORMALIZE(tensor_img)
    np_img = np.transpose(tensor_img.cpu().detach().numpy(), axes=[1, 2, 0])
    return Image.fromarray((np_img * 255).astype(np.uint8))

def create_random_mask(N, L, ratio=0.75):
    len_keep = int(L * (1 - ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1).cuda()

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep].cuda()

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device).cuda()
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return ids_keep, mask, ids_restore

def predict(F, C, img, lbl):
    input_img = classifier_transform()(img).unsqueeze(0).cuda()
    out = C(F(input_img))
    sm = nn.Softmax(dim=1)
    probs = sm(out)
    return probs[0][lbl].item()

def save_mask(vals, input_img, mae):
    vals[np.isnan(vals)] = 0
    vals -= vals.min()
    vals /= vals.max()
    mask = torch.tensor(vals).unsqueeze(0).cuda()
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = mae.unpatchify(mask)[0]
    clr = torch.zeros_like(mask).cuda()
    clr[1, :, :] = 1.0
    out = input_img * (1 - mask) + clr * mask
    to_pil(out).save("mask.png")

def dist(a, b):
    ab_dist = torch.sqrt(((a-b)**2).sum())
    return ab_dist.item()

def patch_dist(org, recon, mask_ids, mae):
    up_ids = list(set(range(mae.patch_embed.num_patches)).difference(set(mask_ids[0].detach().cpu().numpy())))
    up_ids = torch.tensor(up_ids).unsqueeze(0).cuda()
    org_patches = mae.patch_embed(org)
    org_patches = torch.gather(org_patches, dim=1, index=up_ids.unsqueeze(-1).repeat(1, 1, org_patches.shape[2]))
    recon_patches = mae.patch_embed(recon)
    recon_patches = torch.gather(recon_patches, dim=1, index=up_ids.unsqueeze(-1).repeat(1, 1, recon_patches.shape[2]))

    patch_distances = torch.sqrt(((org_patches - recon_patches)**2).sum(2))
    return patch_distances[0].detach().cpu().numpy()

if __name__ == "__main__":
    # Setup
    args = get_args()
    set_random_seed(args.seed)
    cuda_setup(args.gpu_ids)

    # Load models
    F, C, MAE = load_all_models(args.backbone, args.classifier, args.mae)
    MAE = MAE.cuda()

    # load input
    x = load_input(args.img).cuda()
    tmp = x.clone().detach()
    org_feat = F(tmp)
    base_prob = predict(F, C, to_pil(x[0]), args.lbl)
    print(f"Base confidence: {round(base_prob*100, 2)}%")

    num_patches = MAE.patch_embed.num_patches
    counts = np.zeros(num_patches)
    vals = np.zeros(num_patches)
    best_val = 0
    for step in range(args.steps):
        mask_ids, mask, ids_restore = create_random_mask(1, num_patches, ratio=args.ratio)
        out = reconstruct(x, mask_ids, ids_restore, MAE, args.no_mae)
        img = to_pil(out[0])
        criteria = 0
        if args.criteria == 'conf':
            criteria = predict(F, C, img, args.lbl)
            print(f"{step}: Mask confidence: {round(criteria*100, 2)}%")
        elif args.criteria == 'dist':
            criteria = dist(org_feat, F(classifier_transform()(img).unsqueeze(0).cuda()))
            print(f"{step}: Mask dist: {criteria}")
        elif args.criteria == 'patch_dist':
            criteria = patch_dist(x, out, mask_ids, MAE)
            print(f"{step}: Mask patch dist: {criteria.sum()}")

        up_ids = set(range(num_patches)).difference(set(mask_ids[0].detach().cpu().numpy()))
        for i, mask_id in enumerate(up_ids):
            counts[mask_id] += 1
            if args.criteria != "patch_dist":
                vals[mask_id] += criteria
            else:
                vals[mask_id] += criteria[i]
        save_mask(vals/counts, x[0], MAE)

        if best_val < criteria.sum():
            best_val = criteria.sum()
            img.save("test.png")
    


    




    