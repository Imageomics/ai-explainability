import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image

from helpers import parse_xlsx_labels

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--labels', type=str, default="/research/nfs_chao_209/david/Imagenomics/TableS2HoyalCuthilletal2019-Kozak_curated.xlsx")
    parser.add_argument('--dset_root', type=str, default="/research/nfs_chao_209/david/Imagenomics/Cuthill_GoldStandard_Dorsal")
    parser.add_argument('--dset', type=str, default='../datasets/')
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--remove_hybrids', action="store_true", default=False)

    return parser.parse_args()

def get_data(dataset_path, labels_path, remove_hybrids=False):
    id_to_subspecies_map, id_to_hybrid_map = parse_xlsx_labels(labels_path, return_hybrid=True)
    paths = []
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.split(".")[-1] != 'tif': continue
            if f.split("_")[1] != "D": continue #! ONLY DORSAL VIEW
            id = int(f.split('_')[0])
            if id_to_hybrid_map[id] and remove_hybrids: continue #! No Hybrids
            paths.append(os.path.join(root, f))
            labels.append(id_to_subspecies_map[id])
    return paths, labels

def split_data(paths, labels, split):
    subspecies_paths = {}
    for lbl in set(labels):
        subspecies_paths[lbl] = list(map(lambda x: x[0], filter(lambda x: x[1] == lbl, zip(paths, labels))))

    if split == 1.0 or split == 0.0:
        return subspecies_paths
    
    train_data = {}
    test_data = {}
    for subspecies in subspecies_paths:
        if len(subspecies_paths[subspecies]) <= 1: continue # Cannot have train/testing split of one data point
        split_idx = int(len(subspecies_paths[subspecies]) * split)
        train_data[subspecies] = subspecies_paths[subspecies][:split_idx]
        test_data[subspecies] = subspecies_paths[subspecies][split_idx:]
        assert len(set(train_data[subspecies]).intersection(set(test_data[subspecies]))) == 0, "Train & Test sets cannot intersect."
    
    return train_data, test_data

def dist(a, b):
    return np.sqrt(((a - b) ** 2).sum())

def get_background(path, thresh=4):
    img = np.array(Image.open(path))
    h, w = img.shape[:2]
    visited = ["0_{w//2}", "0_0", f"0_{w-1}", f"{h-1}_{w-1}", f"{h-1}_0"]
    queue = [(0, w//2, img[0, w//2]),(0, 0, img[0, 0]), (0, w-1, img[0, w-1]), (h-1, w-1, img[h-1, w-1]), (h-1, 0, img[h-1, 0])]
    background = np.ones_like(img[:, :, 0]).astype(np.uint8)
    background[0,0] = 0
    background[0,w-1] = 0
    background[h-1,w-1] = 0
    background[h-1,0] = 0
    background[0, w//2] = 0
    i = 0

    def get_neighbor_points(row, col):
        points = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0: continue
                points.append((max(min(row + y, img.shape[0]-1), 0), max(min(col + x, img.shape[1]-1), 0)))
        return points
    while len(queue) > 0:
        i += 1
        if i % 1000 == 0:
            Image.fromarray(background * 255).save("in_progress_mask.png")
        row, col, color = queue.pop(0)
        points = get_neighbor_points(row, col)
        for y, x in points:
            c = img[y, x]
            d = dist(color, c)
            if d < thresh:
                background[y, x] = 0
                if f"{y}_{x}" not in visited:
                    queue.append((y, x, c))
                    visited.append(f"{y}_{x}")

    return background

def remove_background(path):
    mask = get_background(path)
    new_image = np.array(Image.open(path))
    new_image[mask == 0] = np.array([210, 210, 210])
    return Image.fromarray(new_image)

def preprocess_dataset(dataset_path, labels_path, destination, split=0.8):
    paths, labels = get_data(dataset_path, labels_path)

    if split == 0.0 or split == 1.0:
        print(split)
        data = split_data(paths, labels, split)
        for subspecies in data:
            cur_dir = os.path.join(destination, subspecies)
            os.makedirs(cur_dir, exist_ok=True)
            for path in data[subspecies]:
                img = remove_background(path)
                name = path.split(os.path.sep)[-1].split(".")[0]
                img.save(os.path.join(cur_dir, f"{name}.png"))
                print(f"Completed {subspecies} preprocessing.")
        return


    train_data, test_data = split_data(paths, labels, split)

    # train
    for subspecies in train_data:
        cur_dir = os.path.join(destination, "train", subspecies)
        os.makedirs(cur_dir, exist_ok=True)
        for path in train_data[subspecies]:
            img = remove_background(path)
            name = path.split(os.path.sep)[-1].split(".")[0]
            img.save(os.path.join(cur_dir, f"{name}.png"))
        print(f"Completed {subspecies} training preprocessing.")
    
    # test
    for subspecies in test_data:
        cur_dir = os.path.join(destination, "test", subspecies)
        os.makedirs(cur_dir, exist_ok=True)
        for path in test_data[subspecies]:
            img = remove_background(path)
            name = path.split(os.path.sep)[-1].split(".")[0]
            img.save(os.path.join(cur_dir, f"{name}.png"))
        print(f"Completed {subspecies} testing preprocessing.")

if __name__ == "__main__":
    args = get_args()
    print(args.dset_root)
    print(args.dset)
    preprocess_dataset(args.dset_root, args.labels, args.dset, args.split)
