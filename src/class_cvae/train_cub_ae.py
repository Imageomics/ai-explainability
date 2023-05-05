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
from options import add_configs

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

def load_models(args):
    in_size = 375
    if args.use_bbox:
        in_size = args.img_size

    num_att_vars = len(cub_z_from_label(torch.tensor([0]), args.root_dset)[0])
    if args.only_recon:
        num_att_vars = None
    
    resnet = None
    if args.use_resnet_encoder:
        resnet = ResNet50(img_ch=3)
    
    iin_ae = IIN_AE_Wrapper(args.depth, args.num_features, in_size, 3, 'an', args.only_recon, \
                            extra_layers=args.extra_layers, num_att_vars=num_att_vars, \
                            add_real_cls_vec=args.add_real_cls_vec, resnet=resnet)
    img_classifier = ResNet50(num_classes=200, img_ch=3)

    if args.continue_checkpoint:
        iin_ae.load_state_dict(torch.load(os.path.join(args.output_dir, args.exp_name, "ae.pt")))
        args.img_classifier = os.path.join(args.output_dir, args.exp_name, "img_classifier.pt")
    elif args.ae:
        iin_ae.load_state_dict(torch.load(args.ae))
        img_classifier.load_state_dict(torch.load(args.img_classifier))
    else:
        img_classifier.load_state_dict(torch.load(args.img_classifier))

    return iin_ae, img_classifier

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--use_bbox', action='store_true', default=False)
    parser.add_argument('--continue_checkpoint', action='store_true', default=False)
    parser.add_argument('--no_scheduler', action='store_true', default=False)
    parser.add_argument('--img_classifier', type=str, default=None)
    parser.add_argument('--add_real_cls_vec', action='store_true', default=False)
    parser.add_argument('--force_hardcode', action='store_true', default=False)
    parser.add_argument('--only_recon', action='store_true', default=False)
    parser.add_argument('--use_resnet_encoder', action='store_true', default=False)
    parser.add_argument('--ae', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--extra_layers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--recon_lambda', type=float, default=1)
    parser.add_argument('--recon_zero_lambda', type=float, default=1)
    parser.add_argument('--cls_lambda', type=float, default=0.1)
    parser.add_argument('--cls_zero_lambda', type=float, default=0.1)
    parser.add_argument('--kl_lambda', type=float, default=0.0001)
    parser.add_argument('--sparcity_lambda', type=float, default=0)
    parser.add_argument('--force_dis_lambda', type=float, default=1)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--exp_name', type=str, default="cub_debug")
    parser.add_argument('--num_features', type=int, default=512)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--port', type=str, default="5001")
    parser.add_argument('--root_dset', type=str, default="/local/scratch/cv_datasets/CUB_200_2011/")
    parser.add_argument('--configs', type=str, default=None)
    return parser.parse_args()

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

def main(rank, world_size, args):
    multi_gpu_setup(rank, world_size, port=args.port)
    ae, img_classifier = load_models(args)
    train_dloader, test_dloader = load_data(args)

    logger = Logger(args.output_dir, args.exp_name)

    unnormalize = None #T.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    trainer = AE_Trainer(ae, img_classifier, lambda x: cub_z_from_label(x, args.root_dset), gpu_id=rank, img_cls_resize_fn=normalize)
    trainer.train(train_dloader, test_dloader, epochs=args.epochs, lr=args.lr, logger=logger, \
                    recon_lambda=args.recon_lambda, recon_zero_lambda=args.recon_zero_lambda, \
                    cls_lambda=args.cls_lambda, cls_zero_lambda=args.cls_zero_lambda, \
                    force_dis_lambda=args.force_dis_lambda, sparcity_lambda=args.sparcity_lambda, \
                    kl_lambda=args.kl_lambda, use_scheduler=(not args.no_scheduler), force_hardcode=args.force_hardcode)
    
    destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    if args.configs is not None:
        add_configs(args, args.configs)
    if args.only_recon:
        args.recon_zero_lambda = 0
        args.cls_lambda = 0
        args.cls_zero_lambda = 0
        args.force_dis_lambda = 0
        args.sparcity_lambda = 0
        args.kl_lambda = 0
        force_hardcode = 0
    assert args.img_classifier is not None or args.continue_checkpoint
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args,), nprocs=world_size)

    