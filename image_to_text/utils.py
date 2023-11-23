import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_plot(name, plot_list):
    plt.plot(list(range(len(plot_list))), plot_list)
    plt.savefig(name)
    print("Plot saved")
    plt.clf()

class DummyModel(nn.Module):
    def forward(self, y):
        return y


def plot_sanity_check_image(epoch, ref_image_path, transformation, tokenizer, model):
    img = Image.open(ref_image_path).convert("RGB")
    img = transformation(img).unsqueeze(0).to(model.device)
    out_caption = model.generate(img, max_new_tokens=50)
    caption = tokenizer.decode(out_caption[0])
    print(f"After {epoch} epochs, the model says: f{caption}")
