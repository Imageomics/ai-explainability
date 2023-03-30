import os
import random
from argparse import ArgumentParser
from turtle import color

from tqdm import tqdm 

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss

from PIL import Image

from models import Res50, VGG16, Classifier
from loggers import Logger
from datasets import ImageFolder

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def train_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        #transforms.RandomCrop((int(resize_size/1.5), int(resize_size/1.5))),
        transforms.Resize((resize_size, resize_size)),
        #transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.0, hue=0),
        transforms.ToTensor(),
        normalize
    ])

def test_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--min_loss", type=float, default=0.1)
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--label_smoothing", action="store_true", default=False)
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--view", type=str, default="D")
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument('--mode', type=str, default='filtered', choices=['filtered', 'original', 'original_nohybrid', 'afhqv2'])

    args = parser.parse_args()
    args.gpus = ",".join(map(lambda x: str(x), args.gpus))
    if args.mode == 'filtered':
        args.train_dataset = '../datasets/train/'
        args.test_dataset = '../datasets/test/'
        args.exp_name = 'filtered_classifier'
    elif args.mode == 'original':
        args.train_dataset = '../datasets/original/train/'
        args.test_dataset = '../datasets/original/test/'
        args.exp_name = 'original_classifier'
    elif args.mode == 'original_nohybrid':
        args.train_dataset = '../datasets/original_nohybrid/train/'
        args.test_dataset = '../datasets/original_nohybrid/test/'
        args.exp_name = 'original_nohybrid_classifier'
    elif args.mode == 'afhqv2':
        args.train_dataset = '../datasets/afhqv2/train/'
        args.test_dataset = '../datasets/afhqv2/test/'
        args.exp_name = 'afhqv2_classifier'


    return args

def rgb_img_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def handle_image_list(image_list):
    if type(image_list) is str:
        if os.path.isdir(image_list):
            lines = []
            subspecies = set()
            for root, dirs, files in os.walk(image_list):
                for f in files:
                    line = os.path.join(root, f)
                    subspecies.add(line.split(os.path.sep)[-2])
                    lines.append(line)
            subspecies = sorted(list(subspecies))
            sub_to_lbl = {}
            for lbl, sub in enumerate(subspecies):
                sub_to_lbl[sub] = lbl
            for i, line in enumerate(lines):
                sub = line.split(os.path.sep)[-2]
                lines[i] += f" {sub_to_lbl[sub]}"

            image_list = lines
        else:
            assert os.path.exists(image_list), "If image_list is a string, then it must be a file that exists"
            try:
                with open(image_list, 'r') as f:
                    image_list = f.readlines()
            except:
                assert False, f"There was an issue reading the image_list file at: {image_list}"
    paths = []
    labels = []
    for line in image_list:
        parts = line.split()
        paths.append(parts[0])
        labels.append(int(parts[1]))

    return paths, labels

class ImageList(Dataset):
    def __init__(self, image_list, view="D", transform=None):
        paths, labels = handle_image_list(image_list)
        self.paths = []
        self.labels = []
        self.path_label_map = {}
        for path, lbl in zip(paths, labels):
            f_view = path.split(os.path.sep)[-1].split(".")[0].split("_")[1]
            if f_view == view:
                self.paths.append(path)
                self.labels.append(lbl)
                self.path_label_map[path] = lbl
        self.transform = transform
        self.loader = rgb_img_loader

    def get_label(self, path):
        if path not in self.path_label_map:
            return None
        
        return self.path_label_map[path]
    
    def get_num_classes(self):
        return max(self.labels) + 1

    def __getitem__(self, index):
        path = self.paths[index]
        lbl = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, lbl, path

    def __len__(self):
        return len(self.paths)

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

    is_butterfly = not (args.mode in ['afhqv2'])

    if is_butterfly:
        train_dset = ImageList(args.train_dataset, view=args.view, transform=train_transform())
        test_dset = ImageList(args.test_dataset, view=args.view, transform=test_transform())
    else:
        train_dset = ImageFolder(args.train_dataset, transform=train_transform())
        test_dset = ImageFolder(args.test_dataset, transform=test_transform())

    
    dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    backbone = None
    if args.net == "resnet":
        backbone = Res50(pretrain=args.pretrain).cuda()
    elif args.net == "vgg":
        backbone = VGG16(pretrain=args.pretrain).cuda()

    classifier = Classifier(backbone.in_features, train_dset.get_num_classes()).cuda()

    if args.freeze:
        optimizer = SGD(list(classifier.parameters()), lr=args.lr)
    else:
        optimizer = SGD(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)
    loss_fn = CrossEntropyLoss()
    if args.label_smoothing:
        loss_fn = CrossEntropyLabelSmooth(train_dset.get_num_classes())
    
    best_train_acc = 0
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
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            logger.save_model(backbone, f"{args.net}_{args.mode}_backbone.pt")
            logger.save_model(classifier, f"{args.net}_{args.mode}_classifier.pt")
        if train_acc >= 1.0 and total_loss < args.min_loss:
            logger.log("Ending Early due to low loss and perfect accuracy")
            logger.save_model(backbone, f"{args.net}_{args.mode}_backbone.pt")
            logger.save_model(classifier, f"{args.net}_{args.mode}_classifier.pt")
            exit()

    logger.save_model(backbone, f"{args.net}_{args.mode}_backbone.pt")
    logger.save_model(classifier, f"{args.net}_{args.mode}_classifier.pt")
