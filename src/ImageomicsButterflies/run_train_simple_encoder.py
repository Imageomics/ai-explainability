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
from loading_helpers import load_models

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def img_transform(resize_size=128):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
    ])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--min_loss", type=float, default=0.1)
    parser.add_argument("--augment_strength", type=float, default=0.1)
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--stylegan_path", type=str, default="stylegan3/training_runs/00000-stylegan3-r-train_128_128-gpus2-batch32-gamma6.6/network-snapshot-005000.pkl")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
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

def load_generator(path):
    G, _, _, _ = load_models(path)
    return G

def generate_images(G, ws):
    w_in = ws.unsqueeze(1).repeat([1, G.mapping.num_ws, 1])
    synth_images = G.synthesis(w_in, noise_mode='const')
    synth_images = (synth_images + 1) * (1/2)
    synth_images = synth_images.clamp(0, 1)
    return synth_images

def calc_loss(G, F, ws, imgs):
    MSELoss = nn.MSELoss().cuda()
    L1Loss = nn.L1Loss().cuda()
    synth_images = generate_images(G, ws)
    dist = MSELoss(imgs, synth_images)
    l1_loss = L1Loss(F(NORMALIZE(imgs)), F(NORMALIZE(synth_images)))

    return dist + l1_loss

def to_out_im(im):
    return np.transpose((im * 255).detach().cpu().numpy().astype(np.uint8), axes=[1, 2, 0])

def save_imgs(imgs, G, ws, logger):
    synth_images = generate_images(G, ws)

    end_img = None
    for im, sim in zip(imgs, synth_images):
        pil_im = to_out_im(im)
        pil_sim = to_out_im(sim)
        row = np.concatenate([pil_im, pil_sim], axis=1)
        if end_img is None:
            end_img = row
        else:
            end_img = np.concatenate([end_img, row], axis=0)
    Image.fromarray(end_img).save(os.path.join(logger.get_save_dir(), "out_img.png"))
      
def get_w_avg(G, batch_size):
    projected_zs = np.random.RandomState(123).randn(10000, G.z_dim)
    projected_zs = torch.from_numpy(projected_zs).cuda()
    projected_zs = projected_zs.mean(0, keepdim=True).repeat([batch_size, 1])
    projected_ws = G.mapping(projected_zs, None)[:, 0, :]
    return projected_ws

if __name__ == "__main__":
    args = get_args()
    setup(args)

    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    # Save Args
    logger.save_json(args.__dict__, "args.json")
    train_dset = ImageFolder(args.train_dataset, transform=img_transform())
    dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dset = ImageFolder(args.test_dataset, transform=img_transform())
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    G = load_generator(args.stylegan_path)

    backbone = None
    F = None
    if args.net == "resnet":
        backbone = Res50(pretrain=args.pretrain).cuda()
        F = Res50(pretrain=True).cuda()
    elif args.net == "vgg":
        backbone = VGG16(pretrain=args.pretrain).cuda()
        F = VGG16(pretrain=True).cuda()

    classifier = Classifier(backbone.in_features, G.w_dim).cuda()

    optimizer = SGD(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)
    loss_fn = CrossEntropyLoss()

    w_avg = get_w_avg(G, args.batch_size)

    for epoch in tqdm(range(args.max_epochs), desc="Training", position=0, ncols=50, colour="green"):
        total_loss = 0
        backbone.train()
        classifier.train()
        for imgs, lbls, _ in tqdm(dataloader, desc="Batch", position=1, ncols=50, leave=False):
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            features = backbone(imgs)
            ws = classifier(features) + w_avg
            loss = calc_loss(G, F, ws, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            save_imgs(imgs, G, ws, logger)


        logger.log(f"Epoch {epoch}")
        logger.log(f"Loss: {total_loss}")

        if epoch % 1000 == 0:
            logger.save_model(backbone, f"{args.net}_encoder_backbone.pt")
            logger.save_model(classifier, f"{args.net}_encoder_last_layer.pt")
            np.savez(f'{logger.get_save_dir()}/w_avg.npz', w_avg=w_avg.cpu().numpy())

    logger.save_model(backbone, f"{args.net}_encoder_backbone.pt")
    logger.save_model(classifier, f"{args.net}_encoder_last_layer.pt")
    np.savez(f'{logger.get_save_dir()}/w_avg.npz', w_avg=w_avg.cpu().numpy())
