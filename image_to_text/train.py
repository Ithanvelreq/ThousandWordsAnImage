import numpy as np
import torch
import torchvision

import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformer import ViT
from torch.utils.data import DataLoader

from data import Cifar, IMBALANCECIFAR10


# TRAIN DATASET
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# TEST DATASET
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]
    _, pred = torch.max(output, dim=-1)
    correct = pred.eq(target).sum() * 1.0
    acc = correct.item() / batch_size
    return acc

def to_tensor(img):
    transform = Compose([
        Resize((32, 32)),
        ToTensor(),
    ])
    x = transform(img)
    return x.unsqueeze(0)

def train_img(input_img_1, input_img_2, epoch=50, model=ViT()):
    losses, acc = np.zeros(epoch), np.zeros(epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ce_loss = nn.BCEWithLogitsLoss()

    for idx in range(epoch):
        data = torch.cat((to_tensor(input_img_1), to_tensor(input_img_2)))
        target = torch.Tensor([[1, 0], [0, 1]])

        out = model(data)
        loss = ce_loss(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses[idx] = np.mean(loss.item())
        acc[idx] = accuracy(out, target)
        print("epoch :", idx, acc[idx], losses[idx])

    return losses, acc

def train(dataset, epoch=40, model=ViT()):
    model.cuda()
    num_class = 10
    cm = torch.zeros(num_class, num_class)

    losses, acc = np.zeros(epoch), np.zeros(epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    ce_loss = nn.CrossEntropyLoss()

    for ep in range(epoch):
        batch_loss, batch_acc = [], []

        for idx, (data, target) in enumerate(dataset):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            out = model(data)
            batch_acc.append(accuracy(out, target))

            loss = ce_loss(out, target)
            batch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"epoch : {ep}, accuracy = {round(batch_acc[idx], 3)}, loss = {round(batch_loss[idx], 3)}")

        losses[ep] = np.min(batch_loss)
        acc[ep] = np.max(batch_acc)

    return losses, acc

# img_1 = Image.open('penguin.jpg')
# img_2 = Image.open('cheval-mustang.jpg')
# losses, acc = train(img_1, img_2)

losses, acc = train(train_loader)

plt.plot(acc)
plt.plot(losses)
plt.show()