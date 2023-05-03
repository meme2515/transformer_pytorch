# QUAK Dataset : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=71268

from model.model import Transformer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchtext.datasets import Multi30k
from util import tokenizer

import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import wandb
import os
import argparse
import shutil
import sys

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 512
MAX_LEN = 256
N_HEADS = 8
N = 6
SRC_VOCAB = tokenizer.de_vocab()
TGT_VOCAB = tokenizer.en_vocab()
LEARNING_RATE = 0.1
BETA1, BETA2 = 0.9, 0.98

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

de_dict = tokenizer.de_dict()
en_dict = tokenizer.en_dict()

wandb.init(
    project="transformer v1",

    config={
        "learning_rate" : LEARNING_RATE,
        "architecture" : "Transformer",
        "dataset" : "Multi30k",
        "beta1" : BETA1,
        "beta2" : BETA2
    }
)

model = Transformer(
    src_vocab=SRC_VOCAB,
    tgt_vocab=TGT_VOCAB,
    d_model=D_MODEL,
    max_len=MAX_LEN,
    n_heads=N_HEADS,
    N=N,
    # device=DEVICE
)

# MPS & CUDA
model.to(DEVICE)
wandb.watch(model, log=None)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

# Adam Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE, 
    betas=(BETA1, BETA2), 
    weight_decay=5e-4,
    eps=1e-9
)

# LR Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                       verbose=True, 
                                                       factor=0.9, 
                                                       patience=10)

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss(ignore_index=1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def idx_to_word(x, vocab):
    words = []
    x = x[:30]
    for i in x:
        word = vocab.get_itos()[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words

# Training Step
def train(epochs, print_every=100):
    model.train()
    
    start = time.time()
    temp = start
    train_loss = 0
    val_loss = 0
    epoch_train_loss = 0
    epoch_val_loss = 0
    train_steps = 0
    val_steps = 0

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=128, collate_fn=tokenizer.collate_fn, shuffle=True)

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=128, collate_fn=tokenizer.collate_fn, shuffle=True)
    
    # model.train(True)
    for epoch in range(epochs):
        i = 0
        # Train Step
        for src, tgt in train_dataloader:
            source = src.transpose(-2, -1).to(DEVICE)
            target = tgt.transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            optimizer.zero_grad()
            preds = model(source, target_input)
            
            loss = criterion(preds.view(-1, preds.size(-1)), target_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            wandb.log({"step train loss": loss})

            train_loss += loss.data
            epoch_train_loss += loss.data

            if (i + 1) % print_every == 0:
                loss_avg = train_loss / print_every
                print("Train : time = %dm, epoch %d, iter = %d, loss = %.10f, lr = %.10f, %ds per %d iters" % \
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, get_lr(optimizer), time.time() - temp, print_every))
                train_loss = 0
                temp = time.time()
                
            i += 1

        train_steps = i + 1
        if (epoch + 1) % 100 == 0:
            torch.save(model, 'checkpoint_ep_%d.pt'%(epoch + 1))
            
        j = 0
        # Validation Step
        with torch.no_grad():
            for src, tgt in val_dataloader:
                source = src.transpose(-2, -1).to(DEVICE)
                target = tgt.transpose(-2, -1).to(DEVICE)

                target_input = target[:, :-1]
                target_output = target[:, 1:].contiguous().view(-1)

                preds = model(source, target_input)
                loss = criterion(preds.view(-1, preds.size(-1)), target_output)

                epoch_val_loss += loss.data

                if (j + 1) % (print_every) == 0:
                    loss_avg = val_loss / print_every
                    print("Validation : time = %dm, epoch %d, iter = %d, loss = %.10f, lr = %.10f, %ds per %d iters" % \
                          ((time.time() - start) // 60, epoch + 1, j + 1, loss_avg, get_lr(optimizer), time.time() - temp, print_every))
                    val_loss = 0
                    temp = time.time()
                j += 1
            val_steps = j + 1

        predicted_sentence = torch.argmax(preds.view(-1, preds.size(-1)), dim=1)
        # print(predicted_sentence.shape)
        predicted_sentence = idx_to_word(predicted_sentence, en_dict)
        original_sentence = idx_to_word(target_output, en_dict)
        print("Predicted Sentence :", predicted_sentence)
        print("Original Sentence :", original_sentence)

        wandb.log({
            "epoch train loss": epoch_train_loss / train_steps,
            "epoch validation loss": epoch_val_loss / val_steps,
        })

        if epoch > 100:
            scheduler.step(epoch_val_loss / val_steps)

        epoch_train_loss = 0
        epoch_val_loss = 0

# System Call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()
    train(epochs=args.epoch)