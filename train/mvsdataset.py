from torch.utils.data import Dataset
import itertools
import os
from pathlib import Path
from numpy.random import default_rng


class CocoDataset(Dataset):
    def __init__(self, path, image_size, seed=0, size=0):
        self.path = path
        self.image_size = image_size
        self.items = []

        mvs_folders = list(Path(self.data_path).glob('*.*'))
        for folder_name in mvs_folders:
            folder_name = os.path.join(folder_name, 'blended_images')
            files = list(Path(folder_name).glob('*.*'))
            view_pairs = itertools.permutations(files, r=2)
            self.items.extend(view_pairs)

        default_rng(seed).shuffle(self.items)
        if size != 0:
            self.items = self.items[:size]

    def __getitem__(self, index):
        file_name1, file_name2 = self.items[index]

    def __len__(self):
        return len(self.items)
