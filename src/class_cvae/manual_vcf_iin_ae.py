import os
import random
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize

from models import Classifier, ImageClassifier, ResNet50
from iin_models.ae import IIN_AE

from utils import save_imgs, create_z_from_label, set_seed, save_tensor_as_graph, calc_img_diff_loss

"""
Goal: Create visual counterfactual
"""

def resize(img):
    return Resize((28, 28))(img)

def load_data():
    transform = Compose([
        Resize((32, 32)),
        ToTensor()
    ])
    test_dset = MNIST(root="data", train=False, transform=transform)

    return test_dset

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--src_lbl', type=int, default=7)
    parser.add_argument('--tgt_lbl', type=int, default=1)
    parser.add_argument('--iin_ae', type=str, default=None)
    parser.add_argument('--img_classifier', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="tmp/")
    parser.add_argument('--exp_name', type=str, default="manual_vcf")
    parser.add_argument('--num_features', type=int, default=20)
    parser.add_argument('--use_resnet', action="store_true", default=False)

    return parser.parse_args()

if __name__ == "__main__":
    set_seed()
    args = get_args()
    test_dset = load_data()

    org_img = None
    for img, lbl in test_dset:
        if lbl == args.src_lbl:
            org_img = img.cuda().unsqueeze(0)
            break

    tgt_lbl = torch.tensor([args.tgt_lbl]).cuda()
    src_lbl = torch.tensor([args.src_lbl]).cuda()

    iin_ae = IIN_AE(4, args.num_features, 32, 1, 'an', False)
    img_classifier = ImageClassifier(10)
    if args.use_resnet:
        img_classifier = ResNet50(num_classes=10)
    iin_ae.load_state_dict(torch.load(args.iin_ae))
    if args.img_classifier is not None:
        img_classifier.load_state_dict(torch.load(args.img_classifier))

    iin_ae.cuda()
    img_classifier.cuda()
    
    iin_ae.eval()
    img_classifier.eval()

    sigmoid = nn.Sigmoid()
    sm = nn.Softmax(dim=1)

    z_src = sigmoid(iin_ae.encode(org_img).sample())
    z_src_manual = z_src.clone()
    z_src_manual[:, :7, 0, 0] = create_z_from_label(src_lbl)[0]
    z_tgt = z_src.clone()
    z_tgt[:, :7, 0, 0] = create_z_from_label(tgt_lbl)[0]

    src_org_img = iin_ae.decode(z_src)
    src_man_img = iin_ae.decode(z_src_manual)
    tgt_man_img = iin_ae.decode(z_tgt)
    
    src_org_conf = sm(img_classifier(resize(src_org_img)))[0][src_lbl.item()]
    src_man_conf = sm(img_classifier(resize(src_man_img)))[0][src_lbl.item()]
    tgt_man_conf = sm(img_classifier(resize(tgt_man_img)))[0][tgt_lbl.item()]

    save_tensor_as_graph(z_src[0, :, 0, 0], os.path.join(args.output_dir, args.exp_name + "_z_chg.png"))
    save_tensor_as_graph(z_src_manual[0, :, 0, 0], os.path.join(args.output_dir, args.exp_name + "_z_edit.png"))
    save_tensor_as_graph(z_tgt[0, :, 0, 0], os.path.join(args.output_dir, args.exp_name + "_z.png"))

    save_imgs(src_man_img, tgt_man_img, [src_man_conf], [tgt_man_conf], os.path.join(args.output_dir, args.exp_name + ".png"))