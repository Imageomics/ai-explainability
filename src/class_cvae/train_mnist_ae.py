import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from trainers.ae_trainer import AE_Trainer
from models import IIN_AE_Wrapper, ResNet50
from logger import Logger
from utils import create_z_from_label
from options import MNIST_VAEGAN_Configs

def resize(img):
    return T.Resize((28, 28))(img)

def load_data(configs):
    train_arr = []
    if configs.apply_rotation:
        train_arr.append(T.RandomRotation(45))
    train_arr.append(T.Resize((configs.img_size, configs.img_size)))
    train_arr.append(T.ToTensor())
    train_transform = T.Compose(train_arr)

    test_transform = T.Compose([
        T.Resize((configs.img_size, configs.img_size)),
        T.ToTensor()
    ])
    train_dset = MNIST(root="data", train=True, transform=train_transform, download=True)
    test_dset = MNIST(root="data", train=False, transform=test_transform)
    train_dloader = DataLoader(train_dset, batch_size=configs.batch_size, shuffle=False, sampler=DistributedSampler(train_dset), num_workers=4, pin_memory=True)
    test_dloader = DataLoader(test_dset, batch_size=configs.batch_size, shuffle=False)

    return train_dloader, test_dloader

def multi_gpu_setup(rank, world_size, port="5001"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def load_models(configs):
    num_att_vars = len(create_z_from_label(torch.tensor([0]))[0])
    if configs.only_recon:
        num_att_vars = None

    configs.num_att_vars = num_att_vars
    
    iin_ae = IIN_AE_Wrapper(configs)
    img_classifier = ResNet50(num_classes=10, img_ch=configs.in_channels)

    if configs.continue_checkpoint:
        iin_ae.load_state_dict(torch.load(os.path.join(configs.output_dir, configs.exp_name, "ae.pt")))
        configs.img_classifier = os.path.join(configs.output_dir, configs.exp_name, "img_classifier.pt")
    elif configs.ae is not None:
        iin_ae.load_state_dict(torch.load(configs.ae))
        img_classifier.load_state_dict(torch.load(configs.img_classifier))
    else:
        img_classifier.load_state_dict(torch.load(configs.img_classifier))

    return iin_ae, img_classifier

def main(rank, world_size, configs):
    multi_gpu_setup(rank, world_size, port=configs.port)
    ae, img_classifier = load_models(configs)
    train_dloader, test_dloader = load_data(configs)

    logger = Logger(configs.output_dir, configs.exp_name)

    trainer = AE_Trainer(ae, img_classifier, create_z_from_label, \
                         gpu_id=rank, img_cls_resize_fn=resize, logger=logger)
    trainer.train(train_dloader, test_dloader, configs)
    
    destroy_process_group()

if __name__ == "__main__":
    configs = MNIST_VAEGAN_Configs()
    if configs.only_recon:
        configs.recon_zero_lambda = 0
        configs.cls_lambda = 0
        configs.cls_zero_lambda = 0
        configs.force_dis_lambda = 0
        configs.sparcity_lambda = 0
        configs.force_hardcode = False
    assert configs.img_classifier is not None or configs.continue_checkpoint
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, configs,), nprocs=world_size)

    