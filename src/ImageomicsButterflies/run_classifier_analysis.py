import os
import random
from argparse import ArgumentParser
from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn import CrossEntropyLoss

from PIL import Image

from models import Res50, VGG16, Classifier
from loggers import Logger
from datasets import ImageFolder

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def test_transform(resize_size=128, crop_size=128, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--net", type=str, choices=["resnet", "vgg"], default="vgg")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone", type=str, default="../saved_models/vgg_backbone.pt")
    parser.add_argument("--classifier", type=str, default="../saved_models/vgg_classifier.pt")
    parser.add_argument("--train_dataset", type=str, default="../datasets/train")
    parser.add_argument("--test_dataset", type=str, default="../datasets/test")
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

def run_analysis(backbone, classifier, dataloader, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(dataloader, desc="Computing Accuracy", position=1, ncols=50, leave=False):
            features = backbone(imgs.cuda())
            out = classifier(features)
            _, preds = torch.max(out, dim=1)
            for pred, lbl in zip(preds.detach().cpu().numpy(), lbls):
                confusion_matrix[lbl, pred] += 1
    class_accuracies = np.zeros(num_classes)
    for lbl in range(num_classes):
        class_accuracies[lbl] = confusion_matrix[lbl, lbl] / confusion_matrix[lbl, :].sum()

    return class_accuracies, confusion_matrix

def save_figures(logger, class_accuracies, confusion_matrix, class_names, training_numbers):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(confusion_matrix)
    ax.set_xticks(range(len(class_accuracies)), labels=class_names, rotation=60)
    ax.set_yticks(range(len(class_accuracies)), labels=class_names)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Labels")
    plt.title("Classifier - Testing")
    plt.savefig(os.path.join(logger.get_save_dir(), "confusion_matrix.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 12))
    class_names, class_accuracies, training_numbers = zip(*sorted(zip(class_names, class_accuracies, training_numbers), key=lambda x: x[2], reverse=True))
    ax.bar(class_names, class_accuracies)
    ax.set_xticks(range(len(class_accuracies)), labels=class_names, rotation=60)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Accuracy")
    ax2 = ax.twinx()
    ax2.plot(class_names, training_numbers, c='red')
    ax2.set_ylabel('# Training Data')
    plt.title("Classifier - Testing")
    plt.savefig(os.path.join(logger.get_save_dir(), "class_accuracies.png"))
    plt.close()

def get_training_numbers(train_dset):
    training_numbers = []
    for i in range(train_dset.get_num_classes()):
        training_numbers.append(len(list(filter(lambda x: x == i, train_dset.labels))))
    
    return training_numbers

if __name__ == "__main__":
    args = get_args()
    setup(args)

    logger = Logger(log_output="file", save_path=args.output, exp_name=args.exp_name)
    # Save Args
    logger.save_json(args.__dict__, "args.json")
    train_dset = ImageFolder(args.train_dataset, transform=test_transform())
    test_dset = ImageFolder(args.test_dataset, transform=test_transform())
    test_dataloader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    backbone = None
    if args.net == "resnet":
        backbone = Res50(pretrain=False).cuda()
    elif args.net == "vgg":
        backbone = VGG16(pretrain=False).cuda()
    backbone.load_state_dict(torch.load(args.backbone))

    classifier = Classifier(backbone.in_features, test_dset.get_num_classes()).cuda()
    classifier.load_state_dict(torch.load(args.classifier))

    backbone.eval()
    classifier.eval()
    class_accuracies, confusion_matrix = run_analysis(backbone, classifier, test_dataloader, num_classes=test_dset.get_num_classes())
    training_numbers = get_training_numbers(train_dset)

    save_figures(logger, class_accuracies, confusion_matrix, test_dset.get_class_names(), training_numbers)
