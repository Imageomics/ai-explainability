import os
import random
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from models import Encoder, Decoder, Classifier, ImageClassifier
from utils import create_img_from_text, MaxQueue, set_seed, save_tensor_as_graph, calc_img_diff_loss

"""
Goal: Create visual counterfactual
"""


def load_data():
    test_dset = MNIST(root="data", train=False, transform=ToTensor())

    return test_dset

def save_imgs(reals, fakes, cls_org, cls_fakes, confs, org_confs, output):
    reals = reals.cpu().detach().numpy()
    fakes = fakes.cpu().detach().numpy()
    cls_org = cls_org.cpu().detach().numpy()
    cls_fakes = cls_fakes.cpu().detach().numpy()

    reals = np.transpose(reals, (0, 2, 3, 1)) * 255
    fakes = np.transpose(fakes, (0, 2, 3, 1)) * 255
    cls_org = np.transpose(cls_org, (0, 2, 3, 1)) * 255
    cls_fakes = np.transpose(cls_fakes, (0, 2, 3, 1)) * 255
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
    for img in cls_org:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)[:, :, :1].astype(np.uint8)
    
    tmp = None
    for img in cls_fakes:
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
    parser.add_argument('--class_decoder', type=str, default=None)
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
    parser.add_argument('--num_class_features', type=int, default=7)

    return parser.parse_args()

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

    encoder = Encoder(args.num_features, use_sigmoid=True)
    decoder = Decoder(args.num_features)
    class_decoder = Decoder(args.num_class_features)
    img_classifier = ImageClassifier(10)
    encoder.load_state_dict(torch.load(args.encoder))
    decoder.load_state_dict(torch.load(args.decoder))
    class_decoder.load_state_dict(torch.load(args.class_decoder))
    img_classifier.load_state_dict(torch.load(args.img_classifier))
 
    encoder.cuda()
    decoder.cuda()
    class_decoder.cuda()
    img_classifier.cuda()
    
    encoder.eval()
    decoder.eval()
    class_decoder.eval()
    img_classifier.eval()

    sm = nn.Softmax(dim=1)

    chg_path = None
    if args.restrict_path:
        size = 10
        a_q = MaxQueue(size=size)
        b_q = MaxQueue(size=size)
        a_c = b_c = 0
        with torch.no_grad():
            for img, lbl in test_dset:
                if lbl == args.src_lbl:
                    z_a = encoder(img.cuda().unsqueeze(0))
                    cls_img = class_decoder(z_a[:, :args.num_class_features])
                    out = img_classifier(cls_img)
                    conf = sm(out)[0, args.src_lbl].item()
                    a_q.add(z_a, conf)
                elif lbl == args.tgt_lbl:
                    z_b = encoder(img.cuda().unsqueeze(0))
                    cls_img = class_decoder(z_b[:, :args.num_class_features])
                    out = img_classifier(cls_img)
                    conf = sm(out)[0, args.tgt_lbl].item()
                    b_q.add(z_b, conf)
        a = a_q.avg_val()
        b = b_q.avg_val()
        chg_path = (b - a)

    min_loss_fn = nn.L1Loss()
    if args.loss_fn == "l2":
        min_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()

    if chg_path is not None:
        z_chg = (torch.ones_like(chg_path).cuda() * -1)
    else:
        z_chg = torch.zeros(args.num_class_features).cuda()
    
    z_chg = z_chg[:args.num_class_features]

    z_chg = z_chg.requires_grad_(True)
    
    params = [z_chg]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    org_confs = None
    if args.no_sample:
        with torch.no_grad():
            z = encoder(org_img)
            #img = decoder(z)

    for epoch in range(args.num_iters):
        org_img_recon = None
        org_cls_img_recon = None

        if not args.no_sample:
            with torch.no_grad():
                z = encoder(org_img)
        if chg_path is not None:
            chg_path = (b - z)
            chg_path = chg_path[:, :args.num_class_features]
            sig_z_chg = nn.Sigmoid()(z_chg) * 1
            delta = (chg_path * sig_z_chg)#.repeat(args.batch_size, 1)
            z_edit = z.clone()
            z_edit[:, :args.num_class_features] += delta
            loss = min_loss_fn(sig_z_chg, torch.zeros_like(z_chg).cuda())
        else:
            z_edit = z.clone()
            z_edit[:, :args.num_class_features] += z_chg.unsqueeze(0).repeat(args.batch_size, 1)
            loss = min_loss_fn(z_chg, torch.zeros_like(z_chg).cuda())

        loss *= args.z_lambda

        imgs_recon = decoder(z_edit)
        cls_img_recon = class_decoder(z_edit[:, :args.num_class_features])
        if args.optimize_on_img_cls:
            if args.reinput:
                imgs_recon = decoder(encoder(imgs_recon))
            out = img_classifier(imgs_recon)
            out_org = img_classifier(org_img)
        else:
            cls_img_org = class_decoder(z[:, :args.num_class_features])
            out_org = img_classifier(cls_img_org)
            
            cls_img_recon = class_decoder(z_edit[:, :args.num_class_features])
            out = img_classifier(cls_img_recon)
        
        confs = sm(out)[:, tgt_lbl[0]]
        if epoch == 0:
            org_confs = confs

        if args.reg_on_image:
            org_img_recon = decoder(z)
            img_diff_loss = calc_img_diff_loss(org_img_recon, imgs_recon, min_loss_fn)
            loss += img_diff_loss * args.img_reg_lambda

        if args.cls_loss_fn == "cel":
            cls_loss = class_loss_fn(out, tgt_lbl.repeat(args.batch_size))
            cls_loss += class_loss_fn(out_org, tgt_lbl.repeat(args.batch_size))
        elif args.cls_loss_fn == "custom":
            logit_diff = out - out_org
            tgt_logit = torch.gather(logit_diff, 1, tgt_lbl.repeat(args.batch_size).reshape(-1, 1))
            src_logit = torch.gather(logit_diff, 1, src_lbl.repeat(args.batch_size).reshape(-1, 1))

            cls_loss = -tgt_logit.mean() + nn.L1Loss()(logit_diff, torch.zeros_like(logit_diff).cuda())

        loss += args.cls_lambda * cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1} | Loss: {loss.item()} | Target Conf: {confs[0].item()}")
        with torch.no_grad():
            if (epoch+1) % 1000 == 0:
                save_tensor_as_graph(z_chg, "z_chg.png")
                save_tensor_as_graph(z_edit[0], "z_edit.png")
                save_tensor_as_graph(z[0], "z.png")
            if org_img_recon is None:
                org_img_recon = decoder(z)

            if org_cls_img_recon is None:
                org_cls_img_recon = class_decoder(z[:, :args.num_class_features])
            save_imgs(org_img_recon, imgs_recon, org_cls_img_recon, cls_img_recon, confs, org_confs, args.output)