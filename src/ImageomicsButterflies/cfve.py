# Counterfactual Visual Explanation Reimplementation for buttefly problem
import random
import math
import os

from argparse import ArgumentParser

from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

from models import Res50, Classifier, VGG16, VGG16_Decoder
from loggers import Logger
from data_tools import NORMALIZE, image_transform, to_tensor, test_image_transform, rgb_img_loader, to_grayscale, cosine_similarity
from datasets import ImageList

def get_size_after_conv(cur_size, module):
    stride = module.stride
    kernel_size = module.kernel_size
    padding = module.padding
    if not isinstance(stride, int):
        stride = stride[0]
    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    if not isinstance(padding, int):
        padding = padding[0]
    
    return ((cur_size + padding*2 - kernel_size) / stride) + 1

def get_size_before_conv(cur_size, module):
    stride = module.stride
    kernel_size = module.kernel_size
    padding = module.padding
    if not isinstance(stride, int):
        stride = stride[0]
    if not isinstance(kernel_size, int):
        kernel_size = kernel_size[0]
    if not isinstance(padding, int):
        padding = padding[0]

    return stride * (cur_size - 1) + kernel_size - padding*2

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dset", type=str, default="../datasets/high_res_butterfly_data_test.txt")
    parser.add_argument("--backbone", type=str, default="../saved_models/vgg_backbone.pt")
    parser.add_argument("--classifier", type=str, default="../saved_models/vgg_classifier.pt")
    
    parser.add_argument("--source_img", type=str, default="/local/scratch/datasets/high_res_butterfly_data_test/lativitta_E/10429045_V_lativitta_E.png")
    parser.add_argument("--distractor_img", type=str, default="/local/scratch/datasets/high_res_butterfly_data_test/aglaope_M/10428111_V_aglaope_M.png")
    parser.add_argument("--feature_dims", nargs="+", type=int, default=[512, 7, 7])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", type=str, default="0")
    
    args = parser.parse_args()
    return args

def get_next_edit(feat_original, feat_target, tgt_lbl, classifier, S=[]):
    highest_conf = 0.0
    edit = [-1, -1]
    sm = nn.Softmax()
    with torch.no_grad():
        for i in range(feat_original.shape[1]):
            if len(list(filter(lambda x: x[0] == i, S))): continue
            for j in range(feat_target.shape[1]):
                if len(list(filter(lambda x: x[1] == j, S))): continue
                tmp = feat_original[:, i].clone()
                feat_original[:, i] = feat_target[:, j]
                out = classifier(feat_original.view(-1).unsqueeze(0))[0]
                conf = sm(out)
                if conf[tgt_lbl] > highest_conf:
                    highest_conf = conf[tgt_lbl]
                    edit = [i, j]
                feat_original[:, i] = tmp
    return edit, highest_conf

def create_guassian_block(size, temp=0.01):
    mid = [size[0] // 2, size[1] // 2]
    rows = np.arange(size[0])
    rows = np.reshape(np.repeat(rows, size[1], axis=0), (size[0], size[1]))
    
    cols = np.reshape(np.arange(size[1]), (1, size[1]))
    cols = np.repeat(cols, size[0], axis=0)

    block = np.exp(-((rows-mid[0])**2 + (cols-mid[1])**2) / 2*temp)
    block /= block.max()

    return block

def visualize_feature(backbone, classifier, img, img_features, img2, img2_features, lbl, S):
    block_size = img.shape[1] // img_features.shape[1]

    for idx, idx2 in S:
        col = idx % img_features.shape[1]
        row = idx // img_features.shape[1]
        window = [row*block_size, (row+1)*block_size, col*block_size, (col+1)*block_size]

        col2 = idx2 % img2_features.shape[1]
        row2 = idx2 // img2_features.shape[1]
        window2 = [row2*block_size, (row2+1)*block_size, col2*block_size, (col2+1)*block_size]

        area_size = [window[1]-window[0], window[3]-window[2]]
        guassian_block = torch.tensor(create_guassian_block(area_size, temp=0.005))
        img2_area = reverse_size(img2[:, window2[0]:window2[1], window2[2]:window2[3]], area_size[0], area_size[1])
        img[:, window[0]:window[1], window[2]:window[3]] = (1-guassian_block)*img[:, window[0]:window[1], window[2]:window[3]] + guassian_block*img2_area
    ToPILImage()(img).save("composite.png")

    with torch.no_grad():
        img = img.cuda()
        out = classifier(backbone(img.unsqueeze(0)))
        sm = nn.Softmax()
        print(sm(out[0])[lbl])

def setup(args):
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def reverse_norm(img):
    return transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])(img)
def reverse_size(img, width=256, height=256):
    return transforms.Resize((width, height))(img)

def main():
    args = get_args()
    setup(args)
    dset = ImageList(args.dset, transform=image_transform())
    #train_dataloader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    backbone = VGG16(pretrain=False).cuda()
    backbone.load_state_dict(torch.load(args.backbone))
    classifier = Classifier(backbone.in_features, dset.get_num_classes()).cuda()
    classifier.load_state_dict(torch.load(args.classifier))
    
    # Select source & distractor img
    args.source_lbl = dset.get_label(args.source_img)
    print(f"Source Label: {args.source_lbl}")
    args.distractor_lbl = dset.get_label(args.distractor_img)
    print(f"Distractor Label: {args.distractor_lbl}")

    source_img = dset.load_img(args.source_img)
    distractor_img = dset.load_img(args.distractor_img)

    ToPILImage()(reverse_norm(source_img)).save("source.png")
    ToPILImage()(reverse_norm(distractor_img)).save("distractor.png")
    
    with torch.no_grad():
        S = []
        source_features = backbone(source_img.unsqueeze(0).cuda())[0]
        distractor_features = backbone(distractor_img.unsqueeze(0).cuda())[0]
        source_features = source_features.view(-1, args.feature_dims[1] * args.feature_dims[2])
        distractor_features = distractor_features.view(-1, args.feature_dims[1] * args.feature_dims[2])

        sm = nn.Softmax()
        max_loops = source_features.shape[1]**2
        for _ in range(max_loops):
            edit, conf = get_next_edit(source_features, distractor_features, args.distractor_lbl, classifier, S)
            print(edit)
            S.append(edit)
            #out = model.predict(source_features.unsqueeze(0))[0]
            source_features[:, edit[0]] = distractor_features[:, edit[1]]
            out = classifier(source_features.view(-1).unsqueeze(0))[0]
            conf = sm(out)
            if conf[args.distractor_lbl] == max(conf): break
        
        print(f"Number of edits: {len(S)}")
        print(f"Confidence: {conf[args.distractor_lbl]}")
        visualize_feature(
            backbone,
            classifier,
            reverse_norm(source_img), 
            source_features.view(source_features.shape[0], int(math.sqrt(source_features.shape[1])), -1), 
            reverse_norm(distractor_img), 
            distractor_features.view(distractor_features.shape[0], int(math.sqrt(distractor_features.shape[1])), -1),
            args.distractor_lbl,
            S)

    



if __name__ == "__main__":
    main()