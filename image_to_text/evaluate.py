import torch
import argparse
import yaml
import torch.nn as nn
import numpy as np
from PIL import Image
from torchmetrics.text import WordErrorRate, BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, VisionEncoderDecoderModel
from Flickr30k import Flick30k
from train import get_default_model
from framework.generate import predict_caption, encode_image


def runCNNRNNModel(img_name):
    photo = encode_image(img_name).reshape((1, 2048))
    caption = predict_caption(photo)
    # print(caption)
    return caption


def evaluate_bleu_rouge(img, target, img_name, rouge, bleu, word_error_rate, tokenizer, model, ViT):
    if ViT:
        out_caption = model.generate(img, max_new_tokens=50)
        caption = tokenizer.decode(out_caption[0])
    else:
        caption = runCNNRNNModel(img_name)
    target = tokenizer.decode(target[0]).replace("<|endoftext|>", "")
    caption = caption.replace("<|endoftext|>", "")
    
    rouge_score = rouge(caption.split(), target.split())["rouge1_fmeasure"].item()
    bleu_score = bleu([target], [caption])
    wer_score = word_error_rate([caption], [target]).item()

    return bleu_score, rouge_score, wer_score


def main(is_ViT):
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
    if is_ViT:
        model_path = "./results/checkpoints/chaozhang_stop"
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    else:
        model = None

    if torch.cuda.is_available() and model is not None:
        model = model.cuda()

    if model is not None: model.eval()

    train_dataset = Flick30k('cuda', args.path_to_dataset, args.path_to_labels, tokenizer)
    val_dataset, test_dataset = train_dataset.split_train_val()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    bleu_score, rouge_score, wer_score = [], [], []
    rouge = ROUGEScore()
    bleu = BLEUScore()
    word_error_rate = WordErrorRate()

    for idx, (data, target, img_name) in enumerate(test_dataloader):
        if idx % 10 == 0:
            print("Idx", idx)

        if idx == 500:
            return (np.mean(bleu_score), np.mean(rouge_score), np.mean(wer_score))

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        bleu_score_i, rouge_score_i, word_error_rate_score_i = evaluate_bleu_rouge(data,
                                                                                   target,
                                                                                   img_name[0],
                                                                                   rouge,
                                                                                   bleu,
                                                                                   word_error_rate,
                                                                                   test_dataloader.dataset.tokenizer,
                                                                                   model,
                                                                                   is_ViT)
        bleu_score.append(bleu_score_i)
        rouge_score.append(rouge_score_i)
        wer_score.append(word_error_rate_score_i)

    return (np.mean(bleu_score), np.mean(rouge_score), np.mean(wer_score))


if __name__ == '__main__':
    bleu, rouge, wer = main(is_ViT=False)
    print(f"Bleu Score is : {bleu}")
    print(f"Rouge Score is : {rouge}")
    print(f"WER Score is : {wer}")
    print("Petit coquing...")