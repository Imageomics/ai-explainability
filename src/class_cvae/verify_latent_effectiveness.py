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
from utils import create_img_from_text

def load_data():
    test_dset = MNIST(root="data", train=False, transform=ToTensor())

    return test_dset

def save_imgs(reals, fakes, confs, org_confs, output):
    reals = reals.cpu().detach().numpy()
    fakes = fakes.cpu().detach().numpy()

    reals = np.transpose(reals, (0, 2, 3, 1)) * 255
    fakes = np.transpose(fakes, (0, 2, 3, 1)) * 255
    diffs = (reals - fakes)
    diffs_pos = np.copy(diffs)
    diffs_pos[diffs_pos < 0] = 0
    diffs_neg = np.copy(diffs)
    diffs_neg[diffs_neg > 0] = 0
    diffs_neg *= -1

    final = None
    for i in range(len(reals)):
        if final is None:
            final = create_img_from_text(reals.shape[1], 14, f"{int(round(org_confs[i].item(), 2)*100)}%")
        else:
            final = np.concatenate((final, create_img_from_text(reals.shape[1], 14, f"{int(round(org_confs[i].item(), 2)*100)}%")), axis=1)
    
    tmp = None
    for i in range(len(reals)):
        if tmp is None:
            tmp = create_img_from_text(reals.shape[1], 14, f"{int(round(confs[i].item(), 2)*100)}%")
        else:
            tmp = np.concatenate((tmp, create_img_from_text(reals.shape[1], 14, f"{int(round(confs[i].item(), 2)*100)}%")), axis=1)
    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in reals:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)
    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)

    tmp = None
    for img in fakes:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in diffs_pos:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in diffs_neg:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)

    Image.fromarray(final[:, :, 0]).save(output)

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


def set_seed(seed=2023):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class MaxQueue:
    def __init__(self, size=10):
        self.size = 10
        self.arr = []

    def add(self, z, conf):
        if len(self.arr) < self.size:
            self.arr.append((z, conf))
        else:
            if len(list(filter(lambda x: x[1] < conf, self.arr))) > 0:
                self.arr.pop(0)
                self.arr.append((z, conf))

        self.arr = sorted(self.arr, key=lambda x: x[1])

    def avg_val(self):
        acc = torch.zeros_like(self.arr[0][0])
        for z, _ in self.arr:
            acc += z
        return acc / len(self.arr)

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