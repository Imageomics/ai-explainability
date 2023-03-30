
import torch
import os
import numpy as np

from torchvision import transforms
from PIL import Image

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
UNNORMALIZE = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])

def image_transform(resize_size=256, crop_size=224, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
    ])

def to_tensor(x):
    return transforms.ToTensor()(x)

def test_image_transform(resize_size=256, crop_size=224, normalize=NORMALIZE):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        normalize
    ])

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

def handle_image_folder(img_dir):
    class_names = set()
    paths = []
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            cname = root.split(os.path.sep)[-1]
            if cname not in class_names:
                class_names.add(cname)
            paths.append(os.path.join(root, f))

    class_names = sorted(list(class_names))
    nm_to_lbl = dict(zip(class_names, range(len(class_names))))

    labels = []
    path_label_map = {}
    for path in paths:
        cname = path.split(os.path.sep)[-2]
        lbl = nm_to_lbl[cname]
        labels.append(lbl)
        path_label_map[path] = lbl

    return paths, labels, path_label_map, class_names

def to_grayscale(img):
    #0.299 R + 0.587 G + 0.114 B
    gray_img = torch.zeros((img.shape[1], img.shape[2]))
    gray_img = gray_img + img[0] * 0.229
    gray_img = gray_img + img[1] * 0.587
    gray_img = gray_img + img[2] * 0.114
    return gray_img

def cosine_similarity(x1, x2):
    top = np.dot(x1, x2)
    bottom = np.linalg.norm(x1) * np.linalg.norm(x2)
    return top / bottom