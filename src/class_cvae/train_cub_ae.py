import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

from trainers.ae_trainer import AE_Trainer
from models import IIN_AE_Wrapper, ResNet50
from datasets import CUB
from logger import Logger
from utils import cub_pad

def load_data(args):
    all_transforms = []
    if not args.use_bbox:
        all_transforms.append(T.Lambda(cub_pad))
        all_transforms.append(T.CenterCrop((375, 375)))
    all_transforms.append(T.RandomRotation(10))
    if args.use_bbox:
        all_transforms.append(T.Resize((256, 256)))
    all_transforms.append(T.ToTensor())
    all_transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    train_transform = T.Compose(all_transforms)

    test_transform_arr = []
    if args.use_bbox:
        test_transform_arr.append(T.Resize((256, 256)))
    else:
        test_transform_arr.append(T.Lambda(cub_pad))
        test_transform_arr.append(T.CenterCrop((375, 375)))

    test_transform_arr.append(T.ToTensor())
    test_transform_arr.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    test_transform = T.Compose(test_transform_arr)

    train_dset = CUB("/local/scratch/cv_datasets/CUB_200_2011/", train=True, bbox=args.use_bbox, transform=train_transform)
    test_dset = CUB("/local/scratch/cv_datasets/CUB_200_2011/", train=False, bbox=args.use_bbox, transform=test_transform)
    train_dloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    return train_dloader, test_dloader

def load_models(args):
    in_size = 375
    if args.use_bbox:
        in_size = 256
    iin_ae = IIN_AE_Wrapper(4, args.num_features, in_size, 3, 'an', False)
    img_classifier = ResNet50(num_classes=200, img_ch=3)

    if args.continue_checkpoint:
        iin_ae.load_state_dict(torch.load(os.path.join(args.output_dir, args.exp_name, "iin_ae.pt")))
        args.img_classifier = os.path.join(args.output_dir, args.exp_name, "img_classifier.pt")

    img_classifier.load_state_dict(torch.load(args.img_classifier))

    return iin_ae, img_classifier

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--use_bbox', action='store_true', default=False)
    parser.add_argument('--continue_checkpoint', action='store_true', default=False)
    parser.add_argument('--img_classifier', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--recon_lambda', type=float, default=1)
    parser.add_argument('--recon_zero_lambda', type=float, default=1)
    parser.add_argument('--cls_lambda', type=float, default=0.1)
    parser.add_argument('--kl_lambda', type=float, default=0.001)
    parser.add_argument('--sparcity_lambda', type=float, default=0.1)
    parser.add_argument('--force_dis_lambda', type=float, default=1)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--exp_name', type=str, default="cub_debug")
    parser.add_argument('--num_features', type=int, default=350)
    return parser.parse_args()

def get_hardcode_cub_latent_map():
    cub_map = {}
    with open("/local/scratch/cv_datasets/CUB_200_2011/attributes/class_attribute_labels_continuous.txt") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            f_vals = np.array(list(map(float, line.split())))
            code = np.zeros_like([f_vals])
            code[0][f_vals[0] >= 50] = 1
            cub_map[i] = code.astype(np.uint8)

    return cub_map

def cub_z_from_label(lbls):
    z_map = get_hardcode_cub_latent_map()

    z = z_map[lbls[0].item()]
    for i in range(1, len(lbls)):
        z = np.concatenate((z, z_map[lbls[i].item()]), axis=0)

    return torch.tensor(z).cuda()

if __name__ == "__main__":
    args = get_args()
    assert args.img_classifier is not None
    ae, img_classifier = load_models(args)
    train_dloader, test_dloader = load_data(args)

    logger = Logger(args.output_dir, args.exp_name)

    unnormalize = T.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])

    trainer = AE_Trainer(ae, img_classifier, cub_z_from_label, unnormalize=unnormalize)
    trainer.train(train_dloader, test_dloader, epochs=args.epochs, lr=args.lr, logger=logger, \
                    recon_lambda=args.recon_lambda, recon_zero_lambda=args.recon_zero_lambda, \
                    cls_lambda=args.cls_lambda, force_dis_lambda=args.force_dis_lambda, \
                    sparcity_lambda=args.sparcity_lambda, kl_lambda=args.kl_lambda)