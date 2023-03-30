import csv
import os
from re import sub
from shutil import copy

hybrids = {}

with open("../datasets/hybrid_data.csv", 'r') as f:
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        if i == 0:
            print(line)
            continue

        hybrid = line[4]
        if hybrid == "N":
            continue
        subspecies = line[3]
        if subspecies not in hybrids:
            hybrids[subspecies] = []
        species = line[2][0].upper()
        view = line[1][0].upper()
        id = line[0]
        hybrids[subspecies].append(f"{id}_{view}_{subspecies}_{species}")

original = "/local/scratch/datasets/high_res_butterfly_data_test_norm/"
target = "/home/carlyn.1/ImagenomicsButterflies/datasets/high_res_butterfly_data_test_norm/"

new_dataset = {}
for root, dirs, paths in os.walk(original):
    for p in paths:
        dp = p.split(".")[0]
        id, view, subspecies, species = dp.split("_")
        if subspecies in hybrids and dp in hybrids[subspecies]:
            print(f"Removed {dp}")
            continue
        if subspecies not in new_dataset:
            new_dataset[subspecies] = []

        new_dataset[subspecies].append(dp)

print(len(new_dataset.keys()))
count = 0
for subspecies in new_dataset:
    print(f"{subspecies}: {len(new_dataset[subspecies])}")
    for dp in new_dataset[subspecies]:
        species = dp.split("_")[3]
        tag = f"{subspecies}_{species}"
        if not os.path.exists(os.path.join(target, tag)):
            os.mkdir(os.path.join(target, tag))
        copy(os.path.join(original, tag, f"{dp}.png"), os.path.join(target, tag, f"{dp}.png"))
        count += 1

print(f"Total dataset size: {count}")

        

        

        