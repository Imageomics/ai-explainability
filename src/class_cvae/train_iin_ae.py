import os
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize, RandomRotation

from PIL import Image

from models import Classifier, ImageClassifier, ResNet50
from iin_models.ae import IIN_AE
from logger import Logger
from lpips.lpips import LPIPS

from utils import init_weights, create_z_from_label

"""
Goal: Train a variational autoencoder with a classification head on the latent space.
"""

def resize(img):
    return Resize((28, 28))(img)

def load_data(batch_size):
    train_transform = Compose([
        RandomRotation(45),
        Resize((32, 32)),
        ToTensor()
    ])
    test_transform = Compose([
        Resize((32, 32)),
        ToTensor()
    ])
    train_dset = MNIST(root="data", train=True, transform=train_transform, download=True)
    test_dset = MNIST(root="data", train=False, transform=test_transform)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)

    return train_dloader, test_dloader

def save_imgs(reals, fakes, output_dir):
    reals = reals.cpu().detach().numpy()
    fakes = fakes.cpu().detach().numpy()

    reals = np.transpose(reals, (0, 2, 3, 1)) * 255
    fakes = np.transpose(fakes, (0, 2, 3, 1)) * 255
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

    final_img = np.concatenate((final, tmp), axis=0)[:, :, 0].astype(np.uint8)

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
    parser.add_argument('--force_disentanglement', action='store_true', default=False)
    parser.add_argument('--use_resnet', action='store_true', default=False)
    parser.add_argument('--add_rotation', action='store_true', default=False)
    parser.add_argument('--continue_checkpoint', action='store_true', default=False)
    parser.add_argument('--pretrain_img_classifier', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--recon_lambda', type=float, default=1)
    parser.add_argument('--recon_zero_lambda', type=float, default=1)
    parser.add_argument('--cls_lambda', type=float, default=0.1)
    parser.add_argument('--img_cls_lambda', type=float, default=0.1)
    parser.add_argument('--normal_lambda', type=float, default=0.001)
    parser.add_argument('--l1_lambda', type=float, default=0.1)
    parser.add_argument('--force_dis_lambda', type=float, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--exp_name', type=str, default="debug")
    parser.add_argument('--num_features', type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_dloader, test_dloader = load_data(args.batch_size)

    classifier = Classifier(args.num_features, 10)

    if args.force_disentanglement:
        classifier = Classifier(7, 10)

    iin_ae = IIN_AE(4, args.num_features, 32, 1, 'an', False)
    img_classifier = ImageClassifier(10)

    if args.continue_checkpoint:
        iin_ae.load_state_dict(torch.load(os.path.join(args.output_dir, args.exp_name, "iin_ae.pt")))
        args.pretrain_img_classifier = os.path.join(args.output_dir, args.exp_name, "img_classifier.pt")
        classifier.load_state_dict(torch.load(os.path.join(args.output_dir, args.exp_name, "classifier.pt")))

    if args.use_resnet:
        img_classifier = ResNet50(num_classes=10)
    total_params = 0
    total_params += sum(p.numel() for p in iin_ae.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    if args.add_img_classifier:
        total_params += sum(p.numel() for p in img_classifier.parameters() if p.requires_grad)

    logger = Logger(args.output_dir, args.exp_name)

    logger.log(f"Total trainable parameters: {total_params}")
    iin_ae.cuda()
    classifier.cuda()

    if not args.continue_checkpoint:
        classifier.apply(init_weights)
        if not args.use_resnet:
            img_classifier.apply(init_weights)

    normal_loss_fn = nn.L1Loss()
    def recon_loss_fn(s, t): 
        return nn.L1Loss()(s, t) + LPIPS()(s, t)
    class_loss_fn = nn.CrossEntropyLoss()
    params = list(iin_ae.parameters())
    if args.add_classifier:
        params += list(classifier.parameters())
    if args.add_img_classifier or args.pretrain_img_classifier:
        if args.pretrain_img_classifier is not None:
            img_classifier.load_state_dict(torch.load(args.pretrain_img_classifier))
        else:
            params += list(img_classifier.parameters())
        img_classifier.cuda()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        losses = {
            "recon" : 0,
            "zero_reg_recon" : 0,
            "latent_cls" : 0,
            "sparcity" : 0,
            "normal" : 0,
            "disentangle" : 0,
            "img_cls" : 0,
            "img_cls_zero" : 0,
            "all" : 0
        }
        total = 0
        correct = 0
        img_correct = 0
        iin_ae.train()
        classifier.train()
        if args.pretrain_img_classifier:
            img_classifier.eval()
        else:
            img_classifier.train()
        
        for imgs, lbls in tqdm(train_dloader, desc="Training"):
            imgs = imgs.cuda()
            lbls = lbls.cuda()

            z_dist = iin_ae.encode(imgs)
            z = nn.Sigmoid()(z_dist.sample())
            z_force = create_z_from_label(lbls).float()
            z_force = z_force.unsqueeze(2).unsqueeze(3)
            imgs_recon_zero_reg = iin_ae.decode(torch.cat((z_force, torch.zeros_like(z[:, 7:, :, :])), 1))
            imgs_recon = iin_ae.decode(z)

            recon_loss = recon_loss_fn(imgs, imgs_recon)
            losses["recon"] += recon_loss.item()
            loss = recon_loss * args.recon_lambda
            
            recon_loss_zero = recon_loss_fn(imgs, imgs_recon_zero_reg)
            losses["zero_reg_recon"] += recon_loss_zero.item()
            loss = recon_loss * args.recon_zero_lambda

            if args.force_disentanglement:
                reg = nn.L1Loss()(z[:, :7], z_force)
                losses["disentangle"] += reg.item()
                loss += reg * args.force_dis_lambda

            normal_loss = z_dist.kl().mean()
            losses["normal"] += normal_loss.item()
            loss += normal_loss * args.normal_lambda

            sparcity_loss = nn.L1Loss()(z, torch.zeros_like(z))
            losses["sparcity"] += sparcity_loss.item()
            loss += sparcity_loss * args.l1_lambda

            total += len(imgs)

            if args.add_classifier:
                if args.force_disentanglement:
                    out = classifier(z[:, :7])
                else:
                    out = classifier(z)

                cls_loss = class_loss_fn(out, lbls)
                losses["latent_cls"] += cls_loss.item()
                loss += cls_loss * args.cls_lambda
                _, preds = torch.max(out, dim=1)

                correct += (preds == lbls).sum().item()

            if args.add_img_classifier or args.pretrain_img_classifier:
                out = img_classifier(resize(imgs_recon))

                img_cls_loss = class_loss_fn(out, lbls)
                losses["img_cls"] += img_cls_loss.item()
                loss += img_cls_loss * args.img_cls_lambda

                _, img_preds = torch.max(out, dim=1)

                img_correct += (img_preds == lbls).sum().item()
                
                out = img_classifier(resize(imgs_recon_zero_reg))

                img_cls_loss_zero = class_loss_fn(out, lbls)
                losses["img_cls_zero"] += img_cls_loss_zero.item()
                loss += img_cls_loss_zero * args.img_cls_lambda


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses["all"] += loss.item()

            save_imgs(imgs, imgs_recon, logger.get_path())

        for key in losses:
            losses[key] = round(losses[key] / len(train_dloader), 4)

        out_string = f"Epoch: {epoch+1} | Total Loss: {losses['all']} | Disentangle Loss: {losses['disentangle']} | Normal Loss: {losses['normal']} | Recon Loss: {losses['recon']} | Zero Recon Loss: {losses['zero_reg_recon']} | Sparsity Loss: {losses['sparcity']}"
        if args.add_classifier:
            out_string += f" | Class Loss: {losses['latent_cls']} | Train Accuracy: {round(correct/total, 4)}"

        if args.add_img_classifier or args.pretrain_img_classifier:
            out_string += f" | Img Class Loss: {losses['img_cls']} | Img Class Zero Loss: {losses['img_cls_zero']} | Image Class Acc: {round(img_correct/total, 4)}"

        logger.log(out_string)

        save_imgs(imgs, imgs_recon, logger.get_path())

        if args.add_classifier:
            total_loss = 0
            correct = 0
            total = 0
            iin_ae.eval()
            classifier.eval()
            with torch.no_grad():
                for imgs, lbls in test_dloader:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                    z_dist = iin_ae.encode(imgs)
                    z = z_dist.sample()

                    if args.force_disentanglement:
                        out = classifier(z[:, :7])
                    else:
                        out = classifier(z)

                    _, preds = torch.max(out, dim=1)

                    correct += (preds == lbls).sum().item()
                    total += len(imgs)

                logger.log(f"Epoch: {epoch+1} | Test Accuracy: {round(correct/total, 4)}")

        torch.save(iin_ae.state_dict(), f"{logger.get_path()}/iin_ae.pt")
        torch.save(classifier.state_dict(), f"{logger.get_path()}/classifier.pt")
        torch.save(img_classifier.state_dict(), f"{logger.get_path()}/img_classifier.pt")
