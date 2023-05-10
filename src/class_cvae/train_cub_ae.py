import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from trainers.ae_trainer import AE_Trainer
from models import IIN_AE_Wrapper, ResNet50
from datasets import CUB
from logger import Logger
from utils import cub_pad
from options import CUB_VAEGAN_Configs

def load_data(args):
    all_transforms = []
    if not args.use_bbox:
        all_transforms.append(T.Lambda(cub_pad))
        all_transforms.append(T.CenterCrop((375, 375)))
    all_transforms.append(T.RandomRotation(10))
    if args.use_bbox:
        all_transforms.append(T.Resize((args.img_size, args.img_size)))
    all_transforms.append(T.ToTensor())
    #all_transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    train_transform = T.Compose(all_transforms)

    test_transform_arr = []
    if args.use_bbox:
        test_transform_arr.append(T.Resize((args.img_size, args.img_size)))
    else:
        test_transform_arr.append(T.Lambda(cub_pad))
        test_transform_arr.append(T.CenterCrop((375, 375)))

    test_transform_arr.append(T.ToTensor())
    #test_transform_arr.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    test_transform = T.Compose(test_transform_arr)

    train_dset = CUB(args.root_dset, train=True, bbox=args.use_bbox, transform=train_transform)
    test_dset = CUB(args.root_dset, train=False, bbox=args.use_bbox, transform=test_transform)
    
    train_dloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(train_dset), num_workers=4, pin_memory=True)
    test_dloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_dloader, test_dloader

def multi_gpu_setup(rank, world_size, port="5001"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def load_models(configs):
    num_att_vars = len(cub_z_from_label(torch.tensor([0]), configs.root_dset)[0])
    if configs.only_recon:
        num_att_vars = None

    configs.num_att_vars = num_att_vars
    
    iin_ae = IIN_AE_Wrapper(configs)
    img_classifier = ResNet50(num_classes=200, img_ch=configs.in_channels)

    if configs.continue_checkpoint:
        iin_ae.load_state_dict(torch.load(os.path.join(configs.output_dir, configs.exp_name, "ae.pt")))
        configs.img_classifier = os.path.join(configs.output_dir, configs.exp_name, "img_classifier.pt")
    elif configs.ae:
        iin_ae.load_state_dict(torch.load(configs.ae))
        img_classifier.load_state_dict(torch.load(configs.img_classifier))
    else:
        img_classifier.load_state_dict(torch.load(configs.img_classifier))

    return iin_ae, img_classifier

def get_hardcode_cub_latent_map(root_dset):
    cub_map = {}
    with open(os.path.join(root_dset, "attributes/class_attribute_labels_continuous.txt")) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            f_vals = np.array(list(map(float, line.split())))
            code = np.zeros_like([f_vals])
            code[0][f_vals[0] >= 50] = 1
            cub_map[i] = code.astype(np.uint8)

    return cub_map

def cub_z_from_label(lbls, root_dset):
    z_map = get_hardcode_cub_latent_map(root_dset)

    z = z_map[lbls[0].item()]
    for i in range(1, len(lbls)):
        z = np.concatenate((z, z_map[lbls[i].item()]), axis=0)

    return torch.tensor(z).cuda()

def main(rank, world_size, configs):
    multi_gpu_setup(rank, world_size, port=configs.port)
    ae, img_classifier = load_models(configs)
    train_dloader, test_dloader = load_data(configs)

    logger = Logger(configs.output_dir, configs.exp_name)

    unnormalize = None #T.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    trainer = AE_Trainer(ae, img_classifier, lambda x: cub_z_from_label(x, configs.root_dset), \
                         gpu_id=rank, img_cls_resize_fn=normalize, logger=logger)
    trainer.train(train_dloader, test_dloader, configs)
    
    destroy_process_group()

if __name__ == "__main__":
    configs = CUB_VAEGAN_Configs()
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

    