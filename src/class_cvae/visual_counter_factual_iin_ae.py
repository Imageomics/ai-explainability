import os
import random
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize

from models import Classifier, ImageClassifier
from iin_models.ae import IIN_AE

from utils import create_img_from_text, save_imgs, MaxQueue, set_seed, save_tensor_as_graph, calc_img_diff_loss

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iin_ae', type=str, default=None)
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
    parser.add_argument('--output_dir', type=str, default="tmp/")
    parser.add_argument('--exp_name', type=str, default="vcf")
    parser.add_argument('--cls_lambda', type=float, default=1.0)
    parser.add_argument('--z_lambda', type=float, default=1.0)
    parser.add_argument('--reg_on_image', action="store_true", default=False)
    parser.add_argument('--img_reg_lambda', type=float, default=1.0)
    parser.add_argument('--num_features', type=int, default=20)

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

    iin_ae = IIN_AE(4, args.num_features, 32, 1, 'an', False)
    classifier = Classifier(args.num_features, 10)
    if args.force_disentanglement:
        classifier = Classifier(7, 10)
    img_classifier = ImageClassifier(10)
    iin_ae.load_state_dict(torch.load(args.iin_ae))
    if args.classifier is not None:
        classifier.load_state_dict(torch.load(args.classifier))
    if args.img_classifier is not None:
        img_classifier.load_state_dict(torch.load(args.img_classifier))

    args.optimize_on_img_cls = args.optimize_on_img_cls or args.classifier is None
    
    iin_ae.cuda()
    classifier.cuda()
    img_classifier.cuda()
    
    iin_ae.eval()
    classifier.eval()
    img_classifier.eval()

    sm = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    chg_path = None
    if args.restrict_path:
        size = 10
        a_q = MaxQueue(size=size)
        b_q = MaxQueue(size=size)
        a_c = b_c = 0
        with torch.no_grad():
            for img, lbl in test_dset:
                if lbl == args.src_lbl:
                    z_a = sigmoid(iin_ae.encode(img.cuda().unsqueeze(0)).sample())
                    out = classifier(z_a)
                    conf = sm(out)[0, args.src_lbl].item()
                    a_q.add(z_a, conf)
                elif lbl == args.tgt_lbl:
                    z_b = sigmoid(iin_ae.encode(img.cuda().unsqueeze(0)).sample())
                    out = classifier(z_b)
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
        z_chg = torch.zeros(20).cuda()
    
    if args.force_disentanglement:
        z_chg = z_chg[:7]

    z_chg = z_chg.requires_grad_(True)
    
    params = [z_chg]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    org_confs = None
    if args.no_sample:
        with torch.no_grad():
            z = sigmoid(iin_ae.encode(org_img).sample())

    for epoch in range(args.num_iters):
        org_img_recon = None

        if not args.no_sample:
            with torch.no_grad():
                z = sigmoid(iin_ae.encode(org_img).sample())
        if chg_path is not None:
            chg_path = (b - z)
            if args.force_disentanglement:
                chg_path = chg_path[:, :7]
            sig_z_chg = nn.Sigmoid()(z_chg) * 1
            delta = (chg_path * sig_z_chg)#.repeat(args.batch_size, 1)
            if args.force_disentanglement:
                z_edit = z.clone()
                z_edit[:, :7] += delta
            else:
                z_edit = z + delta
            loss = min_loss_fn(sig_z_chg, torch.zeros_like(z_chg).cuda())
        else:
            if args.force_disentanglement:
                z_edit = z.clone()
                z_edit[:, :7, 0, 0] += z_chg.unsqueeze(0).repeat(args.batch_size, 1)
            else:
                z_edit = z + z_chg.unsqueeze(0).repeat(args.batch_size, 1)
            loss = min_loss_fn(z_chg, torch.zeros_like(z_chg).cuda())

        loss *= args.z_lambda
        z_edit = torch.clamp(z_edit, 0, 1)
        imgs_recon = iin_ae.decode(z_edit)
        if args.optimize_on_img_cls:
            if args.reinput:
                imgs_recon = iin_ae.decode(sigmoid(iin_ae.encode(imgs_recon).sample()))
            out = img_classifier(resize(imgs_recon))
            out_org = img_classifier(resize(org_img))
        else:
            if args.force_disentanglement:
                out_org = classifier(z[:, :7])
                out = classifier(z_edit[:, :7])
            else:
                out_org = classifier(z)
                out = classifier(z_edit)
        
        confs = sm(out)[:, tgt_lbl[0]]
        if epoch == 0:
            org_confs = confs

        if args.reg_on_image:
            org_img_recon = iin_ae.decode(z)
            img_diff_loss = calc_img_diff_loss(org_img_recon, imgs_recon, min_loss_fn)
            loss += img_diff_loss * args.img_reg_lambda

        if args.cls_loss_fn == "cel":
            cls_loss = class_loss_fn(out, tgt_lbl.repeat(args.batch_size))
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
                if args.force_disentanglement:
                    save_tensor_as_graph(z_chg, os.path.join(args.output_dir, args.exp_name + "_z_chg.png"))
                else:
                    save_tensor_as_graph(z_chg, os.path.join(args.output_dir, args.exp_name + "_z_chg.png"))
                save_tensor_as_graph(z_edit[0, :, 0, 0], os.path.join(args.output_dir, args.exp_name + "_z_edit.png"))
                save_tensor_as_graph(z[0, :, 0, 0], os.path.join(args.output_dir, args.exp_name + "_z_chg.png"))
            if org_img_recon is None:
                org_img_recon = iin_ae.decode(z)

            save_imgs(org_img_recon, imgs_recon, confs, org_confs, os.path.join(args.output_dir, args.exp_name + ".png"))