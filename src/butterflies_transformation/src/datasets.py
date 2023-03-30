import os
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

def collect_paths(path):
    paths = []
    cnames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            cname = root.split(os.path.sep)[-1]
            paths.append(os.path.join(root, f))
            cnames.append(cname)

    return paths, cnames

class CuthillDataset(Dataset):
    def __init__(self, options, train=True, transform=None):
        if train:
            if options.DATA_TO_TRAIN == "all":
                paths, cnames = collect_paths(options.DATASET)
            elif options.DATA_TO_TRAIN in ["train", "test"]:
                paths, cnames = collect_paths(os.path.join(options.DATASET, options.DATA_TO_TRAIN))
            else:
                raise Exception(f"{options.DATA_TO_TRAIN} is not a valid option for CuthilDataset.")
        else:
            paths, cnames = collect_paths(os.path.join(options.DATASET, "test"))

        # SORT and create labels
        unique_cnames = set(cnames)
        unique_cnames = sorted(unique_cnames)
        self.name_lbl_map = dict(zip(unique_cnames, range(len(unique_cnames))))
        self.lbl_map = dict(zip(range(len(unique_cnames)), unique_cnames))

        self.transform = transform
        self.paths = paths
        self.labels = []
        for cname in cnames:
            self.labels.append(self.name_lbl_map[cname])

    def load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def lbl_to_name(self, lbl):
        return self.lbl_map[lbl]
    
    def get_img_by_lbl(self, lbl):
        for i in range(len(self.labels)):
            if self.labels[i] == lbl:
                return self.__getitem__(i)
        return None

    def __getitem__(self, index):
        path = self.paths[index]
        lbl = self.labels[index]
        img = self.load_img(path)

        return img, lbl, path

    def __len__(self):
        return len(self.paths)
    
    def __str__(self):
        out = "======== Cuthill Dataset ========\n"
        out += f"Images: {len(self.paths)}\n"
        img_shape = np.array(self.__getitem__(0)[0]).shape
        out += f"Image shape: {img_shape}\n"
        out += "================================"
        return out


            