import os
import random
from argparse import ArgumentParser

from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image

from models import Res50, VGG16, Classifier
from loggers import Logger

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
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])


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

def setup():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def get_dataset():
    data_path = "../datasets/high_res_butterfly_data_train.txt"
    dataset = ImageList(data_path, transform=test_transform())
    return dataset

def load_model():
    model = Res50(pretrain=False).cuda()
    model.load_state_dict(torch.load("../saved_models/resnet_backbone.pt"))
    return model

def get_features_and_labels(dataset, model):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(dataloader, desc="Extracting Features", position=0, ncols=50, leave=False):
            features = model(imgs.cuda())
            for feat, lbl in zip(features, lbls):
                all_features.append(feat.detach().cpu().numpy())
                all_labels.append(lbl.item())
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    return all_features, all_labels

def tsne_reduce(features):
    return TSNE(random_state=2022, n_components=2).fit_transform(features)    

def visualize(Z, labels):
    fig = plt.figure(figsize=(16, 9))
    for lbl in range(38):
        plt.scatter(Z[labels == lbl][:,0], Z[labels == lbl][:,1], label=lbl)
    plt.title("Dimensionality Reduction Visualization")
    plt.legend()
    plt.savefig("pca_demo.png")

if __name__ == "__main__":
    setup()
    dataset = get_dataset()
    model = load_model()
    features, labels = get_features_and_labels(dataset, model)

    print(f"Dataset Size:")
    print(f"{features.shape[0]} instances")
    print(f"{features.shape[1]} dimensions")

    mean_features = features.mean(1, keepdims=True) # Mean along dimensions
    print(f"Mean features size: {mean_features.shape}")

    centered_features = features - mean_features
    print(f"Centered features size: {centered_features.shape}")

    covariance_features = np.matmul(centered_features.T, centered_features)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_features)
    print(eigenvalues)
    print(f"Eigenvalue shape: {eigenvalues.shape}")
    print(f"Eigenvector shape: {eigenvectors.shape}")

    accumulated_eigenvalues = 0
    threshold = 0.95
    top_K = 0
    for ev in eigenvalues:
        top_K += 1
        accumulated_eigenvalues += ev
        if (accumulated_eigenvalues / eigenvalues.sum()) > threshold:
            break
    print(f"The top {top_K} eigenvectors explain {round(accumulated_eigenvalues / eigenvalues.sum(), 4)*100}% of the variance in the data")
    print(f"The top 2 eigenvectors explain {round(eigenvalues[:2].sum() / eigenvalues.sum(), 4)*100}% of the variance in the data")

    W = eigenvectors[:2]
    print(f"W shape: {W.shape}")
    Z = np.matmul(centered_features, W.T)
    print(f"Z shape: {Z.shape}")

    # Use tsne
    if True:
        Z = tsne_reduce(features)
        

    visualize(Z, labels)
        

