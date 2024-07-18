# Imports
import argparse
import os
import numpy as np
import math
import shutil
from tqdm import tqdm
import pandas as pd
import random

ms = pd.read_csv("updated_master_sheet.csv")
unique_cases = ms.iloc[:,5].dropna()
id_to_patch_map = ms.iloc[:,1:3]

tot_cases = len(unique_cases)
cases_list = []
for c in unique_cases:
    cases_list.append(c)

random.shuffle(cases_list)

train_count = math.ceil(0.92 * tot_cases)
train_cases, test_cases = cases_list[:train_count], cases_list[train_count:]

all_train_paths = []
all_test_paths = []
for i in range(len(id_to_patch_map)):
    map = id_to_patch_map.iloc[i]
    id = map.iloc[0]
    patch = map.iloc[1]
    class_name = patch.split("_")[1]
    path = os.path.join(class_name, patch)
    if id in train_cases:
        all_train_paths.append(path)
    else:
        all_test_paths.append(path)

with open('train_paths.txt', 'w') as f:
    for path in all_train_paths:
        f.write("%s\n" % path)

# Save test paths to a text file
with open('test_paths.txt', 'w') as f:
    for path in all_test_paths:
        f.write("%s\n" % path)

def split_class(class_name, path_prefix, split_path_prefix, train_split, val_split):
    print(f"Splitting class - {class_name}")

    class_path = os.path.join(path_prefix, class_name)

    # Fetching all files inside this class.
    filenames = os.listdir(class_path)

    # Shuffling file names and splitting
    np.random.shuffle(filenames)
    train_len = math.floor(train_split * len(filenames))
    val_len = math.floor(val_split * len(filenames))
    train_names, val_names, test_names = (
        filenames[: train_len],
        filenames[train_len: train_len + val_len],
        filenames[train_len + val_len:]
    )

    # Helper function to copy
    def copy_file_to_split(name, split):
        src_file_path = os.path.join(class_path, name)
        dest_path = os.path.join(split_path_prefix, split, class_name)

        # Creating destination path folders if required.
        os.makedirs(dest_path, exist_ok=True)

        dest_file_path = os.path.join(dest_path, name)
        shutil.copyfile(src_file_path, dest_file_path)

    # Copying files to dedicated folder after splitting
    print("Copying train split")
    for name in tqdm(train_names):
        copy_file_to_split(name, split="train")

    print("Copying val split")
    for name in tqdm(val_names):
        copy_file_to_split(name, split="val")

    print("Copying test split")
    for name in tqdm(test_names):
        copy_file_to_split(name, split="test")

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='This script splits each class of the dataset into train-val-test.'
    )
    parser.add_argument('-p','--path_prefix',
                        help='Relative path to the dataset.',
                        default='./dataset',
                        type=str)
    parser.add_argument('-sp','--split_path_prefix',
                        help='Path to store data after splitting.',
                        default='./data-split',
                        type=str)
    parser.add_argument('-t','--train_split',
                        help='Percentage of train set.',
                        default=0.85,
                        type=float)
    parser.add_argument('-v','--val_split',
                        help='Percentage of validation set.',
                        default=0.07,
                        type=float)
    args = parser.parse_args()

    class_names = os.listdir(args.path_prefix)

    for i, class_name in enumerate(class_names):
        split_class(
            class_name,
            args.path_prefix,
            args.split_path_prefix,
            args.train_split,
            args.val_split
        )