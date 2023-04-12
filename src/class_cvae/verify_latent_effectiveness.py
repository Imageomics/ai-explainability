import os
import random
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from models import Encoder, Decoder, Classifier, ImageClassifier
from utils import create_img_from_text, save_imgs, MaxQueue, set_seed

def load_data():
    test_dset = MNIST(root="data", train=False, transform=ToTensor())

    return test_dset

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--src_lbl', type=int, default=7)
    parser.add_argument('--tgt_lbl', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--encoder', type=str, default=None)
    parser.add_argument('--decoder', type=str, default=None)
    parser.add_argument('--classifier', type=str, default=None)
    parser.add_argument('--img_classifier', type=str, default=None)
    parser.add_argument('--optimize_on_img_cls', action="store_true", default=False)
    parser.add_argument('--optimize_on_both', action="store_true", default=False)
    parser.add_argument('--restrict_path', action="store_true", default=False)
    parser.add_argument('--no_sample', action="store_true", default=False)
    parser.add_argument('--force_disentanglement', action="store_true", default=False)
    parser.add_argument('--reinput', action="store_true", default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['l1', 'l2'])
    parser.add_argument('--cls_loss_fn', type=str, default='cel', choices=['cel', 'custom'])
    parser.add_argument('--output', type=str, default="vcf.png")
    parser.add_argument('--cls_lambda', type=float, default=1.0)
    parser.add_argument('--z_lambda', type=float, default=1.0)
    parser.add_argument('--reg_on_image', action="store_true", default=False)
    parser.add_argument('--img_reg_lambda', type=float, default=1.0)
    parser.add_argument('--num_features', type=int, default=20)

    return parser.parse_args()

def calc_img_diff_loss(org_img_recon, imgs_recon, loss_fn):
    diffs = (org_img_recon - imgs_recon)
    loss = min_loss_fn(diffs, torch.zeros_like(diffs).cuda())
    return loss

def save_z_chg(z_chg, output="z.png"):
    # output is broken
    z = z_chg.detach().cpu().numpy()
    plt.bar(np.arange(len(z)), z)
    plt.savefig(output)
    plt.close()

def get_chg_vec(src_lbl, tgt_lbl):
    z_map = {
        0: np.array([[1, 0, 1, 1, 1, 1, 1]]),
        1: np.array([[0, 0, 0, 0, 1, 0, 1]]),
        2: np.array([[1, 1, 1, 0, 1, 1, 0]]),
        3: np.array([[1, 1, 1, 0, 1, 0, 1]]),
        4: np.array([[0, 1, 0, 1, 1, 0, 1]]),
        5: np.array([[1, 0, 1, 1, 0, 0, 1]]),
        6: np.array([[1, 1, 1, 1, 0, 1, 1]]),
        7: np.array([[1, 0, 0, 0, 1, 0, 1]]),
        8: np.array([[1, 1, 1, 1, 1, 1, 1]]),
        9: np.array([[1, 1, 0, 1, 1, 0, 1]]),
    }

    return torch.tensor(z_map[tgt_lbl.item()] - z_map[src_lbl.item()]).cuda() 


if __name__ == "__main__":
    set_seed()
    args = get_args()
    test_dset = load_data()

    org_img = None
    #TODO make a batch
    i = args.batch_size
    for img, lbl in test_dset:
        if lbl == args.src_lbl:
            i -= 1
            if org_img is None:
                org_img = img.cuda().unsqueeze(0)
            else:
                org_img = torch.cat((org_img, img.unsqueeze(0).cuda()), axis=0)
        if i <= 0: break

    tgt_lbl = torch.tensor([args.tgt_lbl]).cuda()
    src_lbl = torch.tensor([args.src_lbl]).cuda()

    encoder = Encoder(args.num_features, use_sigmoid=args.force_disentanglement)
    decoder = Decoder(args.num_features)
    classifier = Classifier(args.num_features, 10)
    if args.force_disentanglement:
        classifier = Classifier(7, 10)
    img_classifier = ImageClassifier(10)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))
    if args.classifier is not None:
        classifier.load_state_dict(torch.load(args.classifier))
    if args.img_classifier is not None:
        img_classifier.load_state_dict(torch.load(args.img_classifier))

    args.optimize_on_img_cls = args.optimize_on_img_cls or args.classifier is None
    
    encoder.cuda()
    decoder.cuda()
    classifier.cuda()
    img_classifier.cuda()
    
    encoder.eval()
    decoder.eval()
    classifier.eval()
    img_classifier.eval()

    sm = nn.Softmax(dim=1)

    z = encoder(org_img)
    org_img_recon = decoder(z)
    z_edit = z.clone()
    z_chg = get_chg_vec(src_lbl, tgt_lbl)
    z_edit[:, :7] += z_chg


    imgs_recon = decoder(z_edit)
    if args.optimize_on_img_cls:
        out = img_classifier(imgs_recon)
        out_org = img_classifier(org_img)
    else:
        if args.force_disentanglement:
            out_org = classifier(z[:, :7])
            out = classifier(z_edit[:, :7])
        
    org_confs = sm(out)[:, tgt_lbl[0]]
    confs = sm(out)[:, tgt_lbl[0]]

    with torch.no_grad():
        save_z_chg(z_chg[0], "z_chg.png")
        save_z_chg(z_edit[0], "z_edit.png")
        save_z_chg(z[0], "z.png")
        if org_img_recon is None:
            org_img_recon = decoder(z)
        save_imgs(org_img_recon, imgs_recon, confs, org_confs, args.output)