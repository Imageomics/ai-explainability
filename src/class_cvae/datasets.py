import os

import torch
from torch.utils.data import Dataset

from PIL import Image

class CUB(Dataset):
    def __init__(self, root, train=True, bbox=False, transform=None):
        super().__init__()

        self.transform=transform
        self.use_bbox = bbox

        ids = []
        with open(os.path.join(root, "train_test_split.txt")) as f:
            for line in f.readlines():
                id, is_train = line.split()
                if int(is_train) == int(train):
                    ids.append(id)

        self.bboxs = []
        with open(os.path.join(root, "bounding_boxes.txt")) as f:
            for line in f.readlines():
                id, x, y, width, height = line.split()
                if id not in ids: continue
                self.bboxs.append((float(x), float(y), float(x)+float(width), float(y)+float(height)))

        self.img_paths = []
        self.img_lbls = []
        with open(os.path.join(root, "images.txt")) as f:
            for line in f.readlines():
                id, path = line.split()
                if id not in ids: continue
                lbl = int(path.split(".")[0]) - 1
                self.img_paths.append(os.path.join(root, "images", path))
                self.img_lbls.append(lbl)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.img_paths[idx]
        lbl = self.img_lbls[idx]

        img = Image.open(path).convert('RGB')
        if self.use_bbox:
            img = img.crop(self.bboxs[idx])
        if self.transform:
            img = self.transform(img)

        return img, lbl


