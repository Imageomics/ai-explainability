import os
import random
from argparse import ArgumentParser
from turtle import color

from tqdm import tqdm 

import numpy as np

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss

from PIL import Image

from models import Res50, VGG16, Classifier
from loggers import Logger
from datasets import ImageFolder

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def train_transform(resize_size=128, augment=0.1, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=augment, contrast=augment, saturation=augment, hue=0),
        transforms.ToTensor(),
        normalize
    ])

def test_transform(resize_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        normalize
    ])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--min_loss", type=float, default=0.1)
    parser.add_argument("--augment_strength", type=float, default=0.1)
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--train_dataset", type=str, default="../datasets/train")
    parser.add_argument("--test_dataset", type=str, default="../datasets/test")
    parser.add_argument("--view", type=str, default="D")
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")


    args = parser.parse_args()
    args.gpus = ",".join(map(lambda x: str(x), args.gpus))
    return args

def setup(args):
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def compute_accuracy(backbone, classifier, dataloader):
    backbone.eval()
    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(dataloader, desc="Computing Accuracy", position=1, ncols=50, leave=False):
            features = backbone(imgs.cuda())
            out = classifier(features)
            _, preds = torch.max(out, dim=1)
            total += len(lbls)
            correct += (preds.cpu() == lbls).sum()
    return (correct / total).item()


if __name__ == "__main__":
    args = get_args()
    setup(args)

    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    # Save Args
    logger.save_json(args.__dict__, "args.json")
    train_dset = ImageFolder(args.train_dataset, transform=train_transform(augment=args.augment_strength))
    dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dset = ImageFolder(args.test_dataset, transform=test_transform())
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    backbone = None
    if args.net == "resnet":
        backbone = Res50(pretrain=args.pretrain).cuda()
    elif args.net == "vgg":
        backbone = VGG16(pretrain=args.pretrain).cuda()

    classifier = Classifier(backbone.in_features, train_dset.get_num_classes()).cuda()

    optimizer = SGD(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)
    loss_fn = CrossEntropyLoss()

    for epoch in tqdm(range(args.max_epochs), desc="Training", position=0, ncols=50, colour="green"):
        total_loss = 0
        backbone.train()
        classifier.train()
        for imgs, lbls, _ in tqdm(dataloader, desc="Batch", position=1, ncols=50, leave=False):
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            features = backbone(imgs)
            out = classifier(features)
            loss = loss_fn(out, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.log(f"Epoch {epoch}")
        logger.log(f"Loss: {total_loss}")
        train_acc = compute_accuracy(backbone, classifier, dataloader)
        logger.log(f"Train Accuracy: {round(train_acc, 4)*100}%")
        test_acc = compute_accuracy(backbone, classifier, test_dataloader)
        logger.log(f"Test Accuracy: {round(test_acc, 4)*100}%")
        if train_acc >= 1.0 and total_loss < args.min_loss:
            logger.log("Ending Early due to low loss and perfect accuracy")
            logger.save_model(backbone, f"{args.net}_backbone.pt")
            logger.save_model(classifier, f"{args.net}_classifier.pt")
            exit()

    logger.save_model(backbone, f"{args.net}_backbone.pt")
    logger.save_model(classifier, f"{args.net}_classifier.pt")
