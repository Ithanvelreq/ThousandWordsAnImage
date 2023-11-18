import glob
from PIL import Image
import random
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import csv
from torchvision import io, transforms


class Flick30k(Dataset):

    def __init__(self, dev, path_to_dataset, path_to_labels, img_size=(224,224)):
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
        self.transforms = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.5,
                    std=0.5
                ),
                transforms.PILToTensor(),
                transforms.ToTensor()
            ]
        )

        with open(path_to_labels, encoding="utf8") as f:
            labels = f.readlines()

        labels = [label.strip() for label in labels]

        first = True
        for label in labels:
            label = label.replace('"', '')
            if not first and self.label_dict.get(label.split(".")[0], None) is not None:
                captions = self.label_dict.get(label.split(".")[0], None)
                captions.append(label.split(",")[-1])
            elif not first:
                captions = [label.split(",")[-1]]
                self.label_dict[label.split(".")[0]] = captions
            first = False
        self.image_id_list = list(self.label_dict.keys())
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
        label_list = self.label_dict.get(selected_image_id, None)
        label = label_list[random.randint(0, len(label_list))]

        try:
            img_path = os.path.join(self.path_to_dataset , selected_image_id + ".jpg")
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            img_path = os.path.join(self.path_to_dataset, "default.jpg")
            img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def split_train_val(self, val_ratio=0.1, test_ratio=0.1, shuffle=True):
        if shuffle:
            random.shuffle(self.image_id_list)

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
    x = Flick30k('cuda', "./flickr/Images", "./flickr/captions.txt")
    for(a, b) in x:
        print(a)
        print(b)