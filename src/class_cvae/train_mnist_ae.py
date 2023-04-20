import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize, RandomRotation

from trainers.ae_trainer import AE_Trainer
from models import ImageClassifier, IIN_AE_Wrapper, ResNet50
from logger import Logger
from utils import create_z_from_label, init_weights

def resize(img):
    return Resize((28, 28))(img)

def load_data(args):
    train_transform = Compose([
        RandomRotation(45),
        Resize((args.img_size, args.img_size)),
        ToTensor()
    ])
    test_transform = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor()
    ])
    train_dset = MNIST(root="data", train=True, transform=train_transform, download=True)
    test_dset = MNIST(root="data", train=False, transform=test_transform)
    train_dloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    test_dloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    return train_dloader, test_dloader

def load_models(args):
    iin_ae = IIN_AE_Wrapper(4, args.num_features, 32, 1, 'an', False)
    img_classifier = ImageClassifier(10)

    if args.continue_checkpoint:
        iin_ae.load_state_dict(torch.load(os.path.join(args.output_dir, args.exp_name, "iin_ae.pt")))
        args.img_classifier = os.path.join(args.output_dir, args.exp_name, "img_classifier.pt")

    if args.use_resnet:
        img_classifier = ResNet50(num_classes=10)

    if not args.continue_checkpoint:
        if not args.use_resnet:
            img_classifier.apply(init_weights)

    img_classifier.load_state_dict(torch.load(args.img_classifier))

    return iin_ae, img_classifier

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--use_resnet', action='store_true', default=False)
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
    parser.add_argument('--exp_name', type=str, default="debug")
    parser.add_argument('--num_features', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=32)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    assert args.img_classifier is not None
    ae, img_classifier = load_models(args)
    train_dloader, test_dloader = load_data(args)

    logger = Logger(args.output_dir, args.exp_name)

    trainer = AE_Trainer(ae, img_classifier, create_z_from_label, img_cls_resize_fn=resize)
    trainer.train(train_dloader, test_dloader, epochs=args.epochs, lr=args.lr, logger=logger, \
                    recon_lambda=args.recon_lambda, recon_zero_lambda=args.recon_zero_lambda, \
                    cls_lambda=args.cls_lambda, force_dis_lambda=args.force_dis_lambda, \
                    sparcity_lambda=args.sparcity_lambda, kl_lambda=args.kl_lambda)