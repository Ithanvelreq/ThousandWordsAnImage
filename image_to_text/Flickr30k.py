import glob
import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import csv


class Flick30k(Dataset):

    def __init__(self, dev, path_to_dataset, path_to_labels, eager=False):
        self.device = dev
        self.path_to_dataset = path_to_dataset
        self.image_id_list = []
        self.label_dict = {}
        self.train_image_indices = []
        self.val_image_indices = []
        self.test_image_indices = []
        self.train = True
        self.val = False
        self.test = False
        self.eager = eager

        for image_fn in glob.glob(path_to_dataset+"/*"):
            if not "readme" in image_fn:
                self.image_id_list.append(image_fn.split("\\")[-1].split(".")[0])
        with open(path_to_labels, encoding="utf8") as f:
            reader = csv.reader(f, delimiter="|")
            labels = list(reader)
        first = True
        for label in labels:
            if not first and self.label_dict.get(label[0].split(".")[0], None) is not None:
                captions = self.label_dict.get(label[0].split(".")[0], None)
                captions.append(label[1:])
            elif not first:
                captions = [label[1:]]
                self.label_dict[label[0].split(".")[0]] = captions
            first = False
        self.split_train_val()
        print("Done loading!")

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        selected_image_id = None
        if self.train:
            selected_image_id = self.train_image_indices[idx]
        elif self.val:
            selected_image_id = self.val_image_indices[idx]
        else:
            selected_image_id = self.test_image_indices[idx]
        targets = torch.randn(1, 128)
        targets = targets.to(self.device)
        cp = self.real_class_probabilities[idx][None, :].to(self.device)
        targets = torch.cat([targets, cp], 1).float()
        images = self(targets, self.default_mode, self.default_normalization, self.default_noise_interval)
        return images[0], targets[0]

    def split_train_val(self, val_ratio=0.1, test_ratio=0.1, shuffle=True):
        if shuffle : random.shuffle(self.image_id_list)

        val_size = int(len(self) * val_ratio)
        test_size = int(len(self) * test_ratio)
        train_size = len(self) - test_size - val_size

        self.train_image_indices = self.image_id_list[:train_size]
        self.val_image_indices = self.image_id_list[train_size:train_size+val_size]
        self.test_image_indices = self.image_id_list[train_size+val_size:]

    def train(self):
        self.train = True
        self.val = False
        self.test = False

    def val(self):
        self.train = False
        self.val = True
        self.test = False

    def test(self):
        self.train = False
        self.val = False
        self.test = True


if __name__ == '__main__':
    x = Flick30k('cuda', "./Dataset/flickr30k-images", "./Dataset/labels.csv")