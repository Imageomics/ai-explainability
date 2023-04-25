import os
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

def collect_paths(path, only_path=False):
    paths = []
    cnames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if only_path:
                paths.append(os.path.join(root, f))
                continue
            cname = root.split(os.path.sep)[-1]
            cnames.append(cname)
    if only_path:
        return paths

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

        self.num_classes = len(unique_cnames)

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
    


class MyersJiggins(Dataset):
    def __init__(self, options, train=True, transform=None):
        self.id_sub_map = {}
        with open(os.path.join(options.DATASET, "img_subspecies.txt")) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0: continue
                id, sub = line.split(",")
                self.id_sub_map[id] = sub
        if train:
            if options.DATA_TO_TRAIN == "all":
                paths = collect_paths(os.path.join(options.DATASET, f"train_{options.WING_TYPE}"), only_path=True)
                paths += collect_paths(os.path.join(options.DATASET, f"val_{options.WING_TYPE}"), only_path=True)
            elif options.DATA_TO_TRAIN in ["train", "test"]:
                paths = collect_paths(os.path.join(options.DATASET, f"{options.DATA_TO_TRAIN}_{options.WING_TYPE}"), only_path=True)
            else:
                raise Exception(f"{options.DATA_TO_TRAIN} is not a valid option for CuthilDataset.")
        else:
            paths = collect_paths(os.path.join(options.DATASET, f"val_{options.WING_TYPE}"), only_path=True)

        cnames = []
        final_paths = []
        for path in paths:
            id = path.split(os.path.sep)[-1].split("_")[0]
            if id not in self.id_sub_map: continue
            if len(self.id_sub_map[id].split()) > 1: continue # Means hybrid

            final_paths.append(path)
            cnames.append(self.id_sub_map[id])

        paths = final_paths

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

        self.num_classes = len(unique_cnames)

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
        out = "======== Myers-Jiggins Dataset ========\n"
        out += f"Images: {len(self.paths)}\n"
        img_shape = np.array(self.__getitem__(0)[0]).shape
        out += f"Image shape: {img_shape}\n"
        out += "================================"
        return out
            