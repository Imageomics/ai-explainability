
import os
import json
from argparse import ArgumentParser

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

encoder = encoder.cuda()
decoder = decoder.cuda()
classifier = classifier.cuda()

class_loss_fn = torch.nn.CrossEntropyLoss()
reg_loss_fn = torch.nn.L1Loss()
sm = torch.nn.Softmax(dim=1)

def get_mimic_lbl(lbl):
    mimic_map = {
        15 : 3,
        3 : 15,
        
        16 : 8,
        8 : 16,
        
        12 : 7,
        7 : 12,

        14 : 6,
        6 : 14,

        1 : 10,
        10 : 1,

        13 : 5,
        5 : 13,

        9 : 0,
        0 : 9,

        2 : 11,
        11 : 2,

        17 : 4,
        4 : 17
    }

    return mimic_map[lbl]

def save_data(path, img, org_out, org_recon, org_recon_out, cf, cf_out, cf_ref, cf_ref_out, src_lbl, tgt_lbl, dset):
    """
        Save Results

        'Org', 'recon', 
        'cf_ref', 'cf',
        'adds', 'dels'

        Also label
        'Org src conf', 'Recon src conf'
        'Org tgt conf', 'Recon tgt conf'
        'cf src conf', 'cf tgt conf'
    """
    img = tensor_to_numpy_img(img[0])
    org_recon = tensor_to_numpy_img(org_recon[0])
    cf = tensor_to_numpy_img(cf[0])
    cf_ref = tensor_to_numpy_img(cf_ref[0])

    sm = torch.nn.Softmax(dim=1)
    img_src_conf = sm(org_out)[0][src_lbl].item()
    img_tgt_conf = sm(org_out)[0][tgt_lbl].item()
    recon_src_conf = sm(org_recon_out)[0][src_lbl].item()
    recon_tgt_conf = sm(org_recon_out)[0][tgt_lbl].item()
    cf_src_conf = sm(cf_out)[0][src_lbl].item()
    cf_tgt_conf = sm(cf_out)[0][tgt_lbl].item()
    cf_ref_src_conf = sm(cf_ref_out)[0][src_lbl].item()
    cf_ref_tgt_conf = sm(cf_ref_out)[0][tgt_lbl].item()

    diffs = (org_recon - cf)
    diffs_pos = np.copy(diffs)
    diffs_pos[diffs_pos < 0] = 0
    diffs_neg = np.copy(diffs)
    diffs_neg[diffs_neg > 0] = 0
    diffs_neg *= -1

    diffs_pos = np.repeat(np.expand_dims(diffs_pos.sum(2), 2), 3, axis=2)
    diffs_neg = np.repeat(np.expand_dims(diffs_neg.sum(2), 2), 3, axis=2)

    diffs_pos = diffs_pos - diffs_pos.min()
    diffs_pos = ((diffs_pos / diffs_pos.max()) * 255).astype(np.uint8)
    
    diffs_neg = diffs_neg - diffs_neg.min()
    diffs_neg = ((diffs_neg / diffs_neg.max()) * 255).astype(np.uint8)

    img_name = path[0].split(os.sep)[-1].split(".")[0]

    row1 = np.concatenate((img, org_recon), axis=1)
    row2 = np.concatenate((cf_ref, cf), axis=1)
    row3 = np.concatenate((diffs_pos, diffs_neg), axis=1)

    final = np.concatenate((row1, row2, row3), axis=0).astype(np.uint8)

    outdir = os.path.join("../results", f"{dset.lbl_to_name(src_lbl)}_to_{dset.lbl_to_name(tgt_lbl)}")
    if args.do_other:
        outdir = os.path.join("../results_vgg", f"{dset.lbl_to_name(src_lbl)}_to_{dset.lbl_to_name(tgt_lbl)}")
    
    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(final).save(os.path.join(outdir, f"{img_name}.png"))

    stats = {
        "org src conf" : img_src_conf,
        "org tgt conf" : img_tgt_conf,
        "org src conf" : img_src_conf,
        "recon src conf" : recon_src_conf,
        "recon tgt conf" : recon_tgt_conf,
        "cf src conf" : cf_src_conf,
        "cf tgt conf" : cf_tgt_conf,
        "cf ref src conf" : cf_ref_src_conf,
        "cf ref tgt conf" : cf_ref_tgt_conf,
    }

    with open(os.path.join(outdir, f"{img_name}_stats.json"), "w") as f:
        json.dump(stats, f)


for img_num, (img, lbl, path) in enumerate(dloader):
    img = img.cuda()
    lbl = lbl.cuda()

    tgt_lbl = torch.tensor([get_mimic_lbl(lbl[0].item())]).cuda()
    with torch.no_grad():
        z = encoder(img)

    """
    z = z.clone().requires_grad_(True)
    lpips_loss_fn = LPIPS()
    l1_loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([z], lr=args.lr)
    for i in range(args.recon_iterations):
        recon = decoder(z)
        loss = lpips_loss_fn(img, recon) + l1_loss_fn(img, recon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Image {img_num} recon loss: {loss.item()}")
    """


    with torch.no_grad():
        org_out = classifier(img)
        org_recon = decoder(z)
        org_recon_out = classifier(org_recon)
    
    dz = torch.zeros_like(z).requires_grad_(True)

    optimizer = torch.optim.Adam([dz], lr=args.lr)

    for i in range(args.iterations):
        z_cf = z+dz
        cf = decoder(z_cf)
        if args.input_again:
            z_cf = encoder(cf)
            cf = decoder(z_cf)
        cf_out = classifier(cf)

        class_loss = class_loss_fn(cf_out, tgt_lbl)
        reg_loss = reg_loss_fn(z_cf, z)
        img_reg_loss = torch.tensor(0)
        if args.img_reg_lambda > 0:
            img_reg_loss = reg_loss_fn(org_recon, cf)

        loss = class_loss * args.class_lambda + reg_loss * args.reg_lambda + img_reg_loss * args.img_reg_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sm_vs = sm(cf_out)[0]
        src_conf = round(sm_vs[lbl[0].item()].item() * 100, 2)
        tgt_conf = round(sm_vs[tgt_lbl[0].item()].item() * 100, 2)

        print(f"Image {img_num} | Class {dset.lbl_to_name(lbl[0].item())} | Src Conf: {src_conf}% | Tgt Conf: {tgt_conf}% | Class Loss: {class_loss.item()}, Reg Loss: {reg_loss.item()}, Image reg Loss: {img_reg_loss.item()}")
        #if tgt_conf > 99.0: break


    cf_ref, _, _ = dset.get_img_by_lbl(tgt_lbl)
    cf_ref = cf_ref.unsqueeze(0)
    with torch.no_grad():
        cf_ref_out = classifier(cf_ref.cuda())
    save_data(path, img, org_out, org_recon, org_recon_out, cf, cf_out, cf_ref, cf_ref_out, lbl[0].item(), tgt_lbl[0].item(), dset)    
    