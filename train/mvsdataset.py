from torch.utils.data import Dataset
import torch
import itertools
import os
from pathlib import Path
from numpy.random import default_rng
import cv2

from utils import make_query_image


def get_view_pairs(file_name, files):
    view_pairs = []
    with open(file_name) as file:
        lines = file.readlines()
        for line in lines[1:]:
            if len(line) > 3:
                tokens = line.split()
                pair_files = []
                for token in tokens[1::2]:
                    img_id = token.zfill(8)
                    for img_file_name in files:
                        text_name = str(img_file_name)
                        if img_id in text_name and 'mask' not in text_name:
                            pair_files.append(img_file_name)
                pairs = itertools.permutations(pair_files, r=2)
                view_pairs.extend(pairs)
    return view_pairs


class MVSDataset(Dataset):
    def __init__(self, path, image_size, seed=0, epoch_size=0):
        self.path = path
        self.image_size = image_size
        self.items = []
        self.epoch_size = epoch_size

        mvs_folders = list(Path(self.path).glob('*'))
        for folder_name in mvs_folders:
            images_folder = os.path.join(folder_name, 'blended_images')
            files = list(Path(images_folder).glob('*.*'))

            pairs_file = os.path.join(folder_name, 'cams', 'pair.txt')
            if os.path.exists(pairs_file):
                view_pairs = get_view_pairs(pairs_file, files)
                self.items.extend(view_pairs)

        self.rng = default_rng(seed)
        self.rng.shuffle(self.items)
        if epoch_size != 0:
            self.epoch_items = self.items[:epoch_size]

    def reset_epoch(self):
        self.rng.shuffle(self.items)
        if self.epoch_size != 0:
            self.epoch_items = self.items[:self.epoch_size]

    def __getitem__(self, index):
        file_name1, file_name2 = self.items[index]
        img1 = cv2.imread(str(file_name1))
        img1 = make_query_image(img1, self.image_size)
        img2 = cv2.imread(str(file_name2))
        img2 = make_query_image(img2, self.image_size)

        img1 = torch.from_numpy(img1)[None] / 255.0
        img2 = torch.from_numpy(img2)[None] / 255.0
        return img1, img2

    def __len__(self):
        if self.epoch_size != 0:
            return self.epoch_size
        return len(self.items)
