import torch
import argparse
import yaml
import torch.nn as nn
import numpy as np

from PIL import Image
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, VisionEncoderDecoderModel
from Flickr30k import Flick30k
from train import get_default_model

def evaluate_bleu_rouge(img, target, rouge, tokenizer, model):
    out_caption = model.generate(img, max_new_tokens=50)
    caption = tokenizer.decode(out_caption[0]).split()
    target = tokenizer.decode(target[0]).split()

    rouge_score = rouge(caption, target)["rouge1_fmeasure"]
    bleu_score = sentence_bleu(target, caption)

    return bleu_score, rouge_score


def main():
    parser = argparse.ArgumentParser(description='ViT evaluation')
    parser.add_argument('--config', default='./configs/test.yaml')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.unk_token

    model_path = "./results/checkpoints/chaozhang_stop"
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    train_dataset = Flick30k('cuda', args.path_to_dataset, args.path_to_labels, tokenizer)
    val_dataset, test_dataset = train_dataset.split_train_val()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    bleu_score, rouge_score = [], []
    rouge = ROUGEScore()

    for idx, (data, target) in enumerate(test_dataloader):
        if idx % 10 == 0:
            print("Idx", idx)

        if idx % 1000 == 0:
            return (np.mean(bleu_score), np.mean(rouge_score))

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        bleu_score_i, rouge_score_i = evaluate_bleu_rouge(data,
                                                          target,
                                                          rouge,
                                                          test_dataloader.dataset.tokenizer,
                                                          model)
        bleu_score.append(bleu_score_i)
        rouge_score.append(rouge_score_i)

    return (np.mean(bleu_score), np.mean(rouge_score))


if __name__ == '__main__':
    bleu, rouge = main()
    print(f"Bleu Score is : {bleu}")
    print(f"Rouge Score is : {rouge}")
    print("Petit coquing...")