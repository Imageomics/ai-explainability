from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from PIL import Image

from models import Encoder, Decoder, Classifier, ImageClassifier, SimpleEncoder, HandCraftedMNISTDecoder
from logger import Logger
from lpips.lpips import LPIPS
from utils import init_weights

"""
Goal: Train a variational autoencoder with a classification head on the latent space.
"""

def load_data(batch_size):
    train_dset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    test_dset = MNIST(root="data", train=False, transform=ToTensor())
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)

    return train_dloader, test_dloader

def save_imgs(reals, fakes, swaps, output_dir):
    reals = reals.cpu().detach().numpy()
    fakes = fakes.cpu().detach().numpy()
    swaps = swaps.cpu().detach().numpy()

    reals = np.transpose(reals, (0, 2, 3, 1)) * 255
    fakes = np.transpose(fakes, (0, 2, 3, 1)) * 255
    swaps = np.transpose(swaps, (0, 2, 3, 1)) * 255
    final = None
    for img in reals:
        if final is None:
            final = img
        else:
            final = np.concatenate((final, img), axis=1)

    tmp = None
    for img in fakes:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)
    
    tmp = None
    for img in swaps:
        if tmp is None:
            tmp = img
        else:
            tmp = np.concatenate((tmp, img), axis=1)

    final = np.concatenate((final, tmp), axis=0)

    final_img = final[:, :, 0].astype(np.uint8)
    Image.fromarray(final_img).save(f"{output_dir}/recon.png")

def pretrain_img_classifier(classifier, decoder, logger):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    with torch.no_grad():
        z = torch.zeros(128, 7).cuda()
        lbls = torch.zeros(128).long().cuda()
        z_arr = [
            [1, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1]
        ]
        for i, arr in enumerate(z_arr):
            z[i] = torch.tensor(arr).cuda()
            lbls[i] = i
        
        z_str = []
        for item in z_arr:
            tmp = ""
            for v in item:
                tmp += str(v)
            z_str.append(tmp)

        found = 0
        for i in range(128):
            binary = '{0:07b}'.format(i)
            if binary in z_str:
                found += 1
                continue
            arr = []
            for c in binary:
                arr.append(int(c))
            z[i+10-found] = torch.tensor(arr).cuda()
            lbls[i+10-found] = 10

        imgs = decoder(z)
        lbls = lbls.long().cuda()
    class_loss_fn = nn.CrossEntropyLoss()

    for iter in range(100):
        out = classifier(imgs)

        loss = class_loss_fn(out, lbls)

        _, preds = torch.max(out, dim=1)
        print(preds == lbls)
        correct = (preds == lbls).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log(f"Loss: {loss.item()} | Accuracy: {correct/10 * 100}%")

    return classifier

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--add_classifier', action='store_true', default=False)
    parser.add_argument('--add_img_classifier', action='store_true', default=False)
    parser.add_argument('--classify_begin', action='store_true', default=False)
    parser.add_argument('--add_l1', action='store_true', default=False)
    parser.add_argument('--use_handcraft', action='store_true', default=False)
    parser.add_argument('--force_disentanglement', action='store_true', default=False)
    parser.add_argument('--force_cls_reg', action='store_true', default=False)
    parser.add_argument('--pretrain_img_classifier', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--recon_lambda', type=float, default=1)
    parser.add_argument('--cls_lambda', type=float, default=0.1)
    parser.add_argument('--img_cls_lambda', type=float, default=0.1)
    parser.add_argument('--img_swap_lambda', type=float, default=0.01)
    parser.add_argument('--normal_lambda', type=float, default=0.001)
    parser.add_argument('--l1_lambda', type=float, default=0.1)
    parser.add_argument('--force_dis_lambda', type=float, default=1)
    parser.add_argument('--force_cls_reg_lambda', type=float, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--exp_name', type=str, default="debug_swap")
    parser.add_argument('--num_features', type=int, default=20)
    parser.add_argument('--num_class_features', type=int, default=7)
    return parser.parse_args()

def create_z_from_label(lbls):
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

    z = z_map[lbls[0].item()]
    for i in range(1, len(lbls)):
        z = np.concatenate((z, z_map[lbls[i].item()]), axis=0)

    return torch.tensor(z).cuda()

if __name__ == "__main__":
    args = get_args()
    train_dloader, test_dloader = load_data(args.batch_size)

    encoder = Encoder(args.num_features, use_sigmoid=args.force_disentanglement)
    decoder = Decoder(args.num_features)

    if args.force_disentanglement:
        args.num_class_features = 7

    img_classifier = ImageClassifier(10)
    total_params = 0
    total_params += sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in img_classifier.parameters() if p.requires_grad)
    if args.add_img_classifier:
        total_params += sum(p.numel() for p in img_classifier.parameters() if p.requires_grad)

    logger = Logger(args.output_dir, args.exp_name)

    logger.log(f"Total trainable parameters: {total_params}")
    encoder.cuda()
    decoder.cuda()
    img_classifier.cuda()

    encoder.apply(init_weights)
    decoder.apply(init_weights)
    img_classifier.apply(init_weights)

    normal_loss_fn = nn.L1Loss()
    l1_loss_fn = nn.L1Loss()
    lpips_loss_fn = LPIPS()
    class_loss_fn = nn.CrossEntropyLoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    if args.add_img_classifier:
        if args.pretrain_img_classifier is not None:
            img_classifier.load_state_dict(torch.load(args.pretrain_img_classifier))
        else:
            params += list(img_classifier.parameters())
        img_classifier.cuda()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        losses = {
            "recon" : 0,
            "latent_cls" : 0,
            "normal" : 0,
            "img_cls" : 0,
            "img_swap" : 0,
            "all" : 0
        }
        total = 0
        correct = 0
        img_correct = 0
        encoder.train()
        decoder.train()
        if args.pretrain_img_classifier:
            img_classifier.eval()
        else:
            img_classifier.train()
        
        for imgs, lbls in tqdm(train_dloader, desc="Training"):
            imgs = imgs.cuda()
            lbls = lbls.cuda()

            if args.use_handcraft:
                z = encoder(imgs)
            else:
                z, mu, std = encoder(imgs, stats=True)

            imgs_recon = decoder(z)

            recon_loss = lpips_loss_fn(imgs, imgs_recon) + l1_loss_fn(imgs, imgs_recon)
            losses["recon"] += recon_loss.item()
            loss = recon_loss * args.recon_lambda

            if args.force_disentanglement:
                z_force = create_z_from_label(lbls)
                reg = l1_loss_fn(z[:, :7], z_force)
                loss += reg * args.force_dis_lambda

            normal_loss = normal_loss_fn(mu, torch.zeros_like(mu).cuda())
            normal_loss += normal_loss_fn(std, torch.ones_like(mu).cuda())
            losses["normal"] += normal_loss.item()
            loss += normal_loss * args.normal_lambda

            total += len(imgs)

            out = img_classifier(imgs_recon)

            img_cls_loss = class_loss_fn(out, lbls)
            losses["img_cls"] += img_cls_loss.item()
            loss += img_cls_loss * args.img_cls_lambda

            _, img_preds = torch.max(out, dim=1)

            img_correct += (img_preds == lbls).sum().item()

            # SWAPPING

            z_cls = z[:, :args.num_class_features]
            z_ind = z[:, args.num_class_features:]
            z_ind_shuffle = z_ind[torch.randperm(z_ind.shape[0])]

            z_swap = torch.cat((z_cls, z_ind_shuffle), 1)

            swap_recon = decoder(z_swap)

            out = img_classifier(swap_recon)

            img_swap_loss = class_loss_fn(out, lbls) + lpips_loss_fn(swap_recon, imgs)
            losses["img_swap"] += img_swap_loss.item()
            loss += img_swap_loss * args.img_swap_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses["all"] += loss.item()

        for key in losses:
            losses[key] = round(losses[key] / len(train_dloader), 4)

        out_string = f"Epoch: {epoch+1} | Total Loss: {losses['all']} | Normal Loss: {losses['normal']} | Recon Loss: {losses['recon']}"
        out_string += f" | Img Class Loss: {losses['img_cls']} | Image Class Acc: {round(img_correct/total, 4)}"
        out_string += f" | Img Swap Loss: {losses['img_swap']}"

        logger.log(out_string)

        save_imgs(imgs, imgs_recon, swap_recon, logger.get_path())

        """
        
        total_loss = 0
        correct = 0
        total = 0
        encoder.eval()
        with torch.no_grad():
            for imgs, lbls in test_dloader:
                imgs = imgs.cuda()
                lbls = lbls.cuda()

                z = encoder(imgs)

                if args.force_disentanglement:
                    out = classifier(z[:, :7])
                else:
                    out = classifier(z)

                _, preds = torch.max(out, dim=1)

                correct += (preds == lbls).sum().item()
                total += len(imgs)

            logger.log(f"Epoch: {epoch+1} | Test Accuracy: {round(correct/total, 4)}")

        """

        torch.save(encoder.state_dict(), f"{logger.get_path()}/encoder.pt")
        torch.save(decoder.state_dict(), f"{logger.get_path()}/decoder.pt")
        torch.save(img_classifier.state_dict(), f"{logger.get_path()}/img_classifier.pt")
