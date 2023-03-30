import os
import random
from argparse import ArgumentParser
from tqdm import tqdm 

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image

from models import Res50, VGG16, Classifier
from loggers import Logger
from loading_helpers import load_models

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def train_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.RandomCrop((int(resize_size/1.5), int(resize_size/1.5))),
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
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
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--num_classes", type=int, default=34)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone", type=str, default="../saved_models/vgg_backbone_nohybrid_D_norm.pt")
    parser.add_argument("--classifier", type=str, default="../saved_models/vgg_classifier_nohybrid_D_norm.pt")
    parser.add_argument("--dataset", type=str, default="../datasets/high_res_butterfly_data_test_norm.txt")
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
    test_dset = ImageList(args.dataset, view=args.view, transform=test_transform())
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    _, _, F, C = load_models(None, args.backbone, args.classifier)

    test_acc = compute_accuracy(F, C, test_dataloader)
    logger.log(f"Test Accuracy: {round(test_acc, 4)*100}%")
