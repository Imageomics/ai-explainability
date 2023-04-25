"""
Train latent classifier on given VAE model
"""

from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms.functional import pad

from models import ImageClassifier, ResNet50
from logger import Logger
from datasets import CUB

def cub_pad(img):
    pad_size = max(max(img.height, img.width), 500)
    y_to_pad = pad_size - img.height
    x_to_pad = pad_size - img.width

    top_to_pad = y_to_pad // 2
    bottom_to_pad = y_to_pad - top_to_pad
    left_to_pad = x_to_pad // 2
    right_to_pad = x_to_pad - left_to_pad

    return pad(img,
        (left_to_pad, top_to_pad, right_to_pad, bottom_to_pad),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

def load_data(dset, batch_size, use_bbox=False):
    train_transform = T.Compose([
        T.RandomRotation(45),
        T.ToTensor()
    ])

    test_transform = T.Compose([
        T.ToTensor()
    ])

    if dset == "cub":
        flips_and_crop = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
        if not use_bbox:
            flips_and_crop.append(T.RandomCrop((375, 375)))

        train_transform = T.Compose([
            T.Lambda(cub_pad),
            T.RandomOrder(flips_and_crop),
            #T.RandomRotation(10),
            #T.RandomPerspective(distortion_scale=0.25, p=0.5, interpolation=T.InterpolationMode.BILINEAR),
            #T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = T.Compose([
            T.Lambda(cub_pad),
            #T.Resize((256, 256)),
            T.CenterCrop((375, 375)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if dset == "mnist":
        train_dset = MNIST(root="data", train=True, transform=train_transform, download=True)
        test_dset = MNIST(root="data", train=False, transform=test_transform)
    elif dset == "cub":
        train_dset = CUB("/local/scratch/cv_datasets/CUB_200_2011/", train=True, bbox=use_bbox, transform=train_transform)
        test_dset = CUB("/local/scratch/cv_datasets/CUB_200_2011/", train=False, bbox=use_bbox, transform=test_transform)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)

    return train_dloader, test_dloader

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--use_resnet', action="store_true", default=False)
    parser.add_argument('--use_bbox', action="store_true", default=False)
    parser.add_argument('--exp_name', type=str, default="debug_img_classifier")
    parser.add_argument('--dset', type=str, default="mnist", choices=["mnist", "cub"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.dset == "cub":
        args.use_resnet = True
        num_classes = 200
        img_ch = 3
    elif args.dset == "mnist":
        num_classes = 10
        img_ch = 1
    train_dloader, test_dloader = load_data(args.dset, args.batch_size, use_bbox=args.use_bbox)

    classifier = ImageClassifier(num_classes)
    if args.use_resnet:
        classifier = ResNet50(num_classes=num_classes, img_ch=img_ch)
    total_params = 0
    total_params += sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    logger = Logger(args.output_dir, args.exp_name)

    logger.log(f"Total trainable parameters: {total_params}")
    classifier.cuda()

    class_loss_fn = nn.CrossEntropyLoss()
    params = list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(args.epochs):
        losses = []
        correct = 0
        total = 0
        classifier.train()
        for (imgs, lbls) in tqdm(train_dloader):
            imgs = imgs.cuda()
            lbls = lbls.cuda()

            out = classifier(imgs)
            loss = class_loss_fn(out, lbls)

            _, preds = torch.max(out, dim=1)

            correct += (preds == lbls).sum().item()
            total += len(imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        logger.log(f"Epoch: {epoch+1} | Class Loss: {np.mean(losses)} | Train Accuracy: {round(correct/total, 4)}")

        total_loss = 0
        correct = 0
        total = 0
        classifier.eval()
        with torch.no_grad():
            for (imgs, lbls) in tqdm(test_dloader):
                imgs = imgs.cuda()
                lbls = lbls.cuda()

                out = classifier(imgs)

                _, preds = torch.max(out, dim=1)

                correct += (preds == lbls).sum().item()
                total += len(imgs)

            logger.log(f"Epoch: {epoch+1} | Test Accuracy: {round(correct/total, 4)}")

        torch.save(classifier.state_dict(), f"{logger.get_path()}/img_classifier.pt")
