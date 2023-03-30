
import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import numpy as np

from PIL import Image

from options import load_config
from datasets import CuthillDataset
from models import Encoder, Decoder, VGG_Classifier, VGG_Encoder, VGG_Decoder
from tools import tensor_to_numpy_img
from loss.lpips.lpips import LPIPS



parser = ArgumentParser()
parser.add_argument("--recon_iterations", type=int, default=500)
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--class_lambda", type=float, default=1)
parser.add_argument("--reg_lambda", type=float, default=1)
parser.add_argument("--img_reg_lambda", type=float, default=0.0)
parser.add_argument("--do_other", action="store_true", default=False)
parser.add_argument("--input_again", action="store_true", default=False)
args = parser.parse_args()

options = load_config('../configs/cuthill_train.yaml')

dset = CuthillDataset(options, train=True, transform=ToTensor())
dloader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=4)

encoder = Encoder(z_dim=512).eval()
decoder = Decoder(z_dim=512).eval()
classifier = VGG_Classifier(class_num=18).eval()

if not args.do_other:
    encoder.load_state_dict(torch.load("../tmp/encoder_512.pt"))
    decoder.load_state_dict(torch.load("../tmp/decoder_512.pt"))
else:
    encoder = VGG_Encoder(z_dim=512).eval()
    decoder = VGG_Decoder(z_dim=512).eval()
    encoder.load_state_dict(torch.load("../tmp/vgg_encoder_512.pt"))
    decoder.load_state_dict(torch.load("../tmp/vgg_decoder_512.pt"))

classifier.load_state_dict(torch.load("../tmp/classifier.pt"))

#encoder = encoder.cuda()
#decoder = decoder.cuda()
#classifier = classifier.cuda()

class_loss_fn = torch.nn.CrossEntropyLoss()
reg_loss_fn = torch.nn.L1Loss()
sm = torch.nn.Softmax(dim=1)

class_means = {}
class_counts = {}

for img, lbl, path in tqdm(dset, desc="creating hybrids"):

    z = encoder(img.unsqueeze(0))

    if lbl not in class_means:
        class_means[lbl] = torch.zeros_like(z[0])
        class_counts[lbl] = 0
    
    class_means[lbl] += z[0]
    class_counts[lbl] += 1

for lbl in class_means:
    class_means[lbl] /= class_counts[lbl]

for ci in tqdm(class_means, desc="saving hybrids"):
    for ji in class_means:
        if ci == ji: continue
        src_mean = class_means[ci]
        dest_mean = class_means[ji]
        middle_z = src_mean + 0.5*(dest_mean - src_mean)

        hybrid = decoder(middle_z.unsqueeze(0))[0]

        src_name = dset.lbl_to_name(ci)
        dest_name = dset.lbl_to_name(ji)
        Image.fromarray(tensor_to_numpy_img(hybrid)).save(f"../hybrids/{src_name}-{dest_name}-hyrbid.png")
