import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
PATH = '/local/scratch/datasets/butterflies'
LBLS = '../datasets/hybrid_data.csv'

def analyze_background_color():
    data = {}
    with open(LBLS, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0: continue
            cols = line.split(",")
            id = int(cols[0])
            data[id] = {
                "hybrid" : cols[4] == 'Y',
                "subspecies" : cols[3],
                "view" : cols[1]
            }

    name_paths_map = {}
    for root, dirs, files in os.walk(PATH):
        for f in files:
            path = os.path.join(root, f)
            parts = path.split(os.path.sep)[-1].split("_")
            if parts[1] != "D": continue
            id = int(parts[0])
            if data[id]["hybrid"]: continue
            name = data[id]["subspecies"]
            if name not in name_paths_map:
                name_paths_map[name] = []
            name_paths_map[name].append(path)

    for name in name_paths_map:
        avg_color = np.zeros(3)
        for path in name_paths_map[name]:
            avg_color += np.array(Image.open(path))[0,0,:]
        avg_color /= len(name_paths_map[name])
        print(f"Avg Color for {name}: {avg_color}")

if __name__ == "__main__":
    analyze_background_color()

