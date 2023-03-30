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

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from data_tools import NORMALIZE, UNNORMALIZE

def test_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--view", type=str, default="D")
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument('--backbone', type=str, default='../saved_models/vgg_backbone.pt')
    parser.add_argument('--classifier', type=str, default='../saved_models/vgg_classifier.pt')
    parser.add_argument('--img_path', type=str, default='../datasets/train/aglaope/10428242_D_lowres.png')
    parser.add_argument('--img_lbl', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=27)

    args = parser.parse_args()
    args.gpus = ",".join(map(lambda x: str(x), args.gpus))


    return args

def rgb_img_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def setup(args):
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def load_img(path):
    return test_transform()(rgb_img_loader(path))

def save_results(img, hm):
    out_hm = np.transpose(np.tile((hm*255).astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    out_img = np.transpose((img.detach().cpu().numpy()*255).astype(np.uint8), (1, 2, 0))
    result_img = np.concatenate((out_img, out_hm), axis=0)
    Image.fromarray(result_img).save("hm.png")

if __name__ == "__main__":
    args = get_args()
    setup(args)

    img = load_img(args.img_path).unsqueeze(0)
    lbl = args.img_lbl

    backbone = None
    if args.net == "resnet":
        backbone = Res50(pretrain=False).cuda()
    elif args.net == "vgg":
        backbone = VGG16(pretrain=False).cuda()

    backbone.load_state_dict(torch.load(args.backbone))
    classifier = Classifier(backbone.in_features, args.num_classes).cuda()

    model = nn.Sequential(backbone, classifier)
    target_layers = [model[0].layer5[-3]]

    cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(lbl)]
    grayscale_cam = cam(input_tensor=img, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    save_results(UNNORMALIZE(img[0]), grayscale_cam)

