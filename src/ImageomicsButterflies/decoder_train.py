import os
import random
from argparse import ArgumentParser
from turtle import color

from tqdm import tqdm 

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss

from PIL import Image

from models import Res50, VGG16, Classifier, VGG16_Decoder
from loggers import Logger
from data_tools import to_tensor, test_image_transform

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def train_transform(resize_size=256, crop_size=224, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def test_transform(resize_size=256, crop_size=224, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_loss", type=float, default=0.1)
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--train_dataset", type=str, default="../datasets/high_res_butterfly_data_train.txt")
    parser.add_argument("--test_dataset", type=str, default="../datasets/high_res_butterfly_data_test.txt")
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--train_img", type=str, default="/local/scratch/datasets/high_res_butterfly_data_train/aglaope_M/10428279_V_aglaope_M.png")
    parser.add_argument("--test_img", type=str, default="/local/scratch/datasets/high_res_butterfly_data_test/aglaope_M/10428228_V_aglaope_M.png")
    parser.add_argument("--finetune", action="store_true", default=False)


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
    path_label_map = {}
    for line in image_list:
        parts = line.split()
        path_label_map[parts[0]] = int(parts[1])
        paths.append(parts[0])
        labels.append(int(parts[1]))

    return paths, labels, path_label_map

class ImageList(Dataset):
    def __init__(self, image_list, transform=None):
        self.paths, self.labels, self.path_label_map = handle_image_list(image_list)
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

def compute_loss(backbone, decoder, dataloader):
    loss_fn = MSELoss()
    backbone.eval()
    decoder.eval()
    total = 0
    reverse_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(dataloader, desc="Computing Loss", position=1, ncols=50, leave=False):
            features = backbone(imgs.cuda())
            out = decoder(features)
            loss = loss_fn(reverse_norm(imgs.cuda()), out)
            total += loss.item()
    return total

def save_imgs(backbone, decoder, args, logger):
    backbone.eval()
    decoder.eval()
    train_img = test_transform()(rgb_img_loader(args.train_img)).cuda()
    test_img = test_transform()(rgb_img_loader(args.test_img)).cuda()
    reverse_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    reverse_size = transforms.Resize((256, 256))

    train_img_out = reverse_size(transforms.ToPILImage()(reverse_norm(train_img)))
    test_img_out = reverse_size(transforms.ToPILImage()(reverse_norm(test_img)))

    with torch.no_grad():
        out = decoder(backbone(train_img.unsqueeze(0)))[0]
        train_recon_out = reverse_size(transforms.ToPILImage()(reverse_norm(out)))
        out = decoder(backbone(test_img.unsqueeze(0)))[0]
        test_recon_out = reverse_size(transforms.ToPILImage()(reverse_norm(out)))

    fig = plt.figure(figsize=(12, 8))
    # Train Original
    fig.add_subplot(2, 2, 1)
    plt.imshow(train_img_out)
    plt.axis('off')
    plt.title('Original Train Image')

    # Train Transformed
    fig.add_subplot(2, 2, 2)
    plt.imshow(train_recon_out)
    plt.axis('off')
    plt.title('Transformed Train Image')

    # Test Original
    fig.add_subplot(2, 2, 3)
    plt.imshow(test_img_out)
    plt.axis('off')
    plt.title('Original Test Image')

    # Test Transformed
    fig.add_subplot(2, 2, 4)
    plt.imshow(test_recon_out)
    plt.axis('off')
    plt.title('Transformed Test Image')

    plt.savefig(os.path.join(logger.get_save_dir(), f"reconstruction.png"))
    plt.close()


if __name__ == "__main__":
    args = get_args()
    setup(args)

    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    # Save Args
    logger.save_json(args.__dict__, "args.json")
    train_dset = ImageList(args.train_dataset, transform=train_transform())
    dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dset = ImageList(args.test_dataset, transform=train_transform())
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    backbone = VGG16(pretrain=True).cuda()
    backbone.load_state_dict(torch.load(args.backbone))
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    decoder = VGG16_Decoder().cuda()
    if args.decoder is not None:
        decoder.load_state_dict(torch.load(args.decoder))

    optimizer = Adam(decoder.parameters(), lr=args.lr)
    if args.finetune:
        optimizer = Adam(list(backbone.parameters()) + list(decoder.parameters()), lr=args.lr)

    loss_fn = MSELoss()

    reverse_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])

    for epoch in tqdm(range(args.max_epochs), desc="Training", position=0, ncols=50, colour="green"):
        total_loss = 0
        decoder.train()
        backbone.eval()
        if args.finetune and epoch > args.warmup_epochs:
            backbone.train()

        if args.finetune and epoch == args.warmup_epochs:
            logger.log("Warmup completed")
            for param in backbone.parameters():
                param.requires_grad = True
            optimizer = Adam(list(backbone.parameters()) + list(decoder.parameters()), lr=args.lr)

        for imgs, lbls, _ in tqdm(dataloader, desc="Batch", position=1, ncols=50, leave=False):
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            features = backbone(imgs)
            reconstruction = decoder(features)
            imgs = reverse_norm(imgs)
            loss = loss_fn(imgs, reconstruction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.log(f"Epoch {epoch}")
        logger.log(f"Loss: {total_loss}")

        #train_loss = compute_loss(backbone, decoder, dataloader)
        #logger.log(f"Train Loss: {train_loss}")
        #test_loss = compute_loss(backbone, decoder, test_dataloader)
        #logger.log(f"Test Loss: {test_loss}")

        save_imgs(backbone, decoder, args, logger)
    
    #torch.save(backbone.state_dict(), f"{args.net}_backbone.pt")
    torch.save(decoder.state_dict(), f"{args.net}_decoder.pt")