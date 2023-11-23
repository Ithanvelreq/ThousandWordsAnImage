import glob
from transformers import GPT2Tokenizer
from PIL import Image
import random
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import csv
from torchvision import io, transforms


class Flick30k(Dataset):

    def __init__(self, dev, path_to_dataset, path_to_labels, tokenizer, img_size=(224, 224), df=None):
        self.device = dev
        self.path_to_dataset = path_to_dataset
        self.max_length = 50
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.5,
                    std=0.5
                ),
            ]
        )
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(path_to_labels, delimiter="|")
            self.df.rename(columns={" comment_number": "comment_number", " comment": "comment"}, inplace=True)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.comment.iloc[idx]
        image_name = self.df.image_name.iloc[idx]

        try:
            img_path = os.path.join(self.path_to_dataset, image_name)
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            img_path = os.path.join(self.path_to_dataset, "default.jpg")
            img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        caption = self.tokenizer(caption, padding='max_length', max_length=self.max_length).input_ids
        # it would seem that not all captions end up being the same length. Some have length 50 and others 62
        return img, torch.Tensor(caption).long()

    def split_train_val(self, val_ratio=0.1, test_ratio=0.1):
        val_size = int(len(self) * val_ratio)
        test_size = int(len(self) * test_ratio)

        df_test = self.df.iloc[:test_size]
        df_val = self.df.iloc[test_size:test_size+val_size]
        self.df = self.df.iloc[test_size+val_size:]

        dataset_val = Flick30k(self.device, self.path_to_dataset, self.path_to_dataset, self.tokenizer, df=df_val)
        dataset_test = Flick30k(self.device, self.path_to_dataset, self.path_to_dataset, self.tokenizer, df=df_test)
        return dataset_val, dataset_test



if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    x = Flick30k('cuda', "./flickr/Images", "./flickr/labels.csv", tokenizer)
    _, a = x.split_train_val()
    for(a, b) in x:
        print(a)
        print(b)