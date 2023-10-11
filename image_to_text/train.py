import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from transformer import ViT
import matplotlib.pyplot as plt


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]
    _, pred = torch.max(output, dim=-1)
    correct = pred.eq(target).sum() * 1.0
    acc = correct.item() / batch_size
    return acc

def to_tensor(img):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    x = transform(img)
    return x.unsqueeze(0)

def train(input_img_1, input_img_2, epoch=50, model=ViT()):
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

img_1 = Image.open('penguin.jpg')
img_2 = Image.open('cheval-mustang.jpg')
losses, acc = train(img_1, img_2)

plt.plot(acc)
plt.plot(losses)
plt.show()