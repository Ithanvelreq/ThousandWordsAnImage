import os
import argparse
import time
import yaml
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
from utils import AverageMeter, save_plot, DummyModel, plot_sanity_check_image, get_default_model

parser = argparse.ArgumentParser(description='Visual Transformer for Image Captioning training loop')
parser.add_argument('--config', default='./configs/test.yaml')
BATCH_SIZE = 64
losses_list = []

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


def train(epoch, train_data, model, optimizer):
    iter_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    for idx, (data, target) in enumerate(train_data):
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        pred_caption = model(pixel_values=data, labels=target)
        # loss = pred_caption["loss"]
        loss = pred_caption.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), train_data.batch_size)
        # We still need to compute accuracy.
        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Train: Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                   .format(epoch, idx, len(train_data),
                           iter_time=iter_time, loss=losses))
            plot_sanity_check_image(idx, args.ref_image_path, train_data.dataset.transform,
                                    train_data.dataset.tokenizer, model)
    losses_list.append([losses.avg.item()])


def validate(epoch, val_data, model):
    iter_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    # evaluation loop
    for idx, (data, target) in enumerate(val_data):
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.inference_mode():
            pred_caption = model(pixel_values=data, labels=target)
            loss = pred_caption.loss

        losses.update(loss, val_data.batch_size)
        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Validation: Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, idx, len(val_data),
                          iter_time=iter_time, loss=losses))

    losses_list[-1].append(losses.avg.item())


def main():
    if not os.path.exists('./results'):
        os.makedirs('./results')
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.unk_token
    if args.model == 'DummyModel':
        model = DummyModel()
    elif args.model == 'ViT':
        model = ImageCaptioningModel()
    else:
        model = get_default_model(tokenizer)
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    # TODO: Add a scheduler
    train_dataset = Flick30k('cuda', args.path_to_dataset, args.path_to_labels, tokenizer)
    val_dataset, test_dataset = train_dataset.split_train_val()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    try:
        for epoch in range(args.epochs):
            train(epoch, train_dataloader, model, optimizer)
            validate(epoch, val_dataloader, model)
            plot_sanity_check_image(epoch, args.ref_image_path, train_dataset.transform, tokenizer, model)
            save_plot("./results/training_curve.png", losses_list)

            if epoch % args.save_rate == 0:
                if not os.path.exists('./results/checkpoints'):
                    os.makedirs('./results/checkpoints')
                torch.save(model.state_dict(), './results/checkpoints/' +
                           args.model.lower() + '_' + str(epoch) + '.pth')
                print("Model saved successfully")
        # test(test_dataloader, model, criterion)
    except KeyboardInterrupt:
        save_plot("./results/training_curve.png", losses_list)
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(model.state_dict(), './checkpoints/' +
                   args.model.lower() + '_stop.pth')
        print("Model saved successfully")


if __name__ == '__main__':
    main()
