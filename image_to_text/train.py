import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor
from transformer import ImageCaptioningModel
from Flickr30k import Flick30k
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from data import Cifar, IMBALANCECIFAR10

BATCH_SIZE = 64

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


def train(train_data, epoch=40, model=ImageCaptioningModel()):
    model.cuda()

    losses, acc = np.zeros(epoch), np.zeros(epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    ce_loss = nn.CrossEntropyLoss()

    for ep in range(epoch):
        batch_loss, batch_acc = [], []

        for idx, (data, target) in enumerate(train_data):
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

            if idx % 1 == 0:
                print(f"epoch : {ep}, accuracy = {round(batch_acc[idx], 3)}, loss = {round(batch_loss[idx], 3)}")

        losses[ep] = np.min(batch_loss)
        acc[ep] = np.max(batch_acc)

    return losses, acc


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_dataset = Flick30k('cuda', "./flickr/Images", "./flickr/labels.csv", tokenizer)
    val_dataset, test_dataset = train_dataset.split_train_val()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    losses, acc = train(train_dataloader)
    plt.plot(acc)
    plt.plot(losses)
    plt.show()
