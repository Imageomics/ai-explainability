import os
import random
from argparse import ArgumentParser

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

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def KNN_transform(resize_size=128):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
    ])

def train_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
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
    parser.add_argument("--num_classes", type=int, default=34)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--train_dataset", type=str, default="../datasets/high_res_butterfly_data_train.txt")
    parser.add_argument("--test_dataset", type=str, default="../datasets/high_res_butterfly_data_test.txt")
    parser.add_argument("--view", type=str, default="D")
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")


    args = parser.parse_args()
    args.gpus = ",".join(map(lambda x: str(x), args.gpus))
    return args

def rgb_img_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def handle_image_list(image_list):
    if type(image_list) is str:
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
            features = backbone(imgs.cuda()[:, :, 0, 0])
            out = classifier(features)
            _, preds = torch.max(out, dim=1)
            total += len(lbls)
            correct += (preds.cpu() == lbls).sum()
    return (correct / total).item()

def run_KNN(train_dset, test_dset, K=5):
    train_pixels = []
    train_lbls = []
    for img, lbl, _ in train_dset:
        px = img[:, 0,0]
        train_pixels.append(list(px.detach().cpu().numpy()))
        train_lbls.append(lbl)
    test_pixels = []
    test_lbls = []
    for img, lbl, _ in test_dset:
        px = img[:, 0,0]
        test_pixels.append(list(px.detach().cpu().numpy()))
        test_lbls.append(lbl)

    train_pixels = np.array(train_pixels)
    train_pixel_lbls = np.array(train_lbls)
    test_pixels = np.array(test_pixels)

    def dist(query_px, dset):
        diff = dset - query_px
        px_dist = (diff ** 2).sum(1)
        return px_dist

    def mode(x):
        vals, counts = np.unique(x, return_counts=True)
        mode_val = np.argwhere(counts==np.max(counts))
        return vals[mode_val].flatten().tolist()

    correct = 0
    for px, lbl in zip(test_pixels, test_lbls):
        query_px = np.tile(px, (train_pixels.shape[0], 1))
        px_dist = dist(query_px, train_pixels)
        lbls = train_pixel_lbls[np.argsort(px_dist)[:K]]
        pred = mode(lbls)
        if lbl in pred:
            correct += 1

    return correct / len(test_lbls)

def run_MLP(train_dl, test_dl):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = 10
            self.layer = nn.Linear(3, self.in_features)

        def forward(self, x, compute_z=False):
            return self.layer(x).view(x.size(0), -1)
        
    backbone = MLP().cuda()
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
            features = backbone(imgs[:, :, 0, 0])
            out = classifier(features)
            loss = loss_fn(out, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"Loss: {total_loss}")
            train_acc = compute_accuracy(backbone, classifier, dataloader)
            print(f"Train Accuracy: {round(train_acc, 4)*100}%")
            test_acc = compute_accuracy(backbone, classifier, test_dataloader)
            print(f"Test Accuracy: {round(test_acc, 4)*100}%")

        test_acc = compute_accuracy(backbone, classifier, test_dataloader)
        return test_acc

if __name__ == "__main__":
    args = get_args()
    setup(args)

    train_dset = ImageList(args.train_dataset, view=args.view, transform=train_transform())
    dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dset = ImageList(args.test_dataset, view=args.view, transform=test_transform())
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    acc = run_MLP(dataloader, test_dataloader)
    print(f"MLP Accuracy: {round(acc, 4)*100}%")

    train_dset = ImageList(args.train_dataset, view=args.view, transform=KNN_transform())
    test_dset = ImageList(args.test_dataset, view=args.view, transform=KNN_transform())

    # Nearest Neighbor
    knn_1_acc = run_KNN(train_dset, test_dset, K=1)
    knn_5_acc = run_KNN(train_dset, test_dset, K=5)
    knn_10_acc = run_KNN(train_dset, test_dset, K=10)

    print(f"KNN 1 Accuracy: {round(knn_1_acc, 4)*100}%")
    print(f"KNN 5 Accuracy: {round(knn_5_acc, 4)*100}%")
    print(f"KNN 10 Accuracy: {round(knn_10_acc, 4)*100}%")