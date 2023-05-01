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

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 512
MAX_LEN = 256
N_HEADS = 8
N = 6
SRC_VOCAB = tokenizer.de_vocab()
TGT_VOCAB = tokenizer.en_vocab()
LEARNING_RATE = 0.0001
BETA1, BETA2 = 0.9, 0.98

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

wandb.init(
    project="transformer v1",

    config={
        "learning_rate" : LEARNING_RATE,
        "architecture" : "Transformer",
        "dataset" : "QUAK-H",
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
)

# MPS & CUDA
model.to(DEVICE)
wandb.watch(model, log=None)

# Model initialization
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Adam Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE, 
    betas=(BETA1, BETA2), 
    eps=1e-9
)

# LR Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Training Step
def train(epochs, print_every=10):
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
    train_dataloader = DataLoader(train_iter, batch_size=64, collate_fn=tokenizer.collate_fn)

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=64, collate_fn=tokenizer.collate_fn)
    
    for epoch in range(epochs):
        i = 0
        # Train Step
        for src, tgt in train_dataloader:
            source = src.transpose(-2, -1).to(DEVICE)
            target = tgt.transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            model.train(True)
            preds = model(source, target_input)
            optimizer.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=1)
            loss.backward()
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

        if (epoch + 1) % 10 == 0:
            torch.save(model, 'checkpoint_ep_%d.pt'%(epoch + 1))
        
        j = 0
        # Validation Step
        for src, tgt in val_dataloader:
            source = src.transpose(-2, -1).to(DEVICE)
            target = tgt.transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            model.train(False)
            preds = model(source, target_input)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=1)

            epoch_val_loss += loss.data

            if (j + 1) % print_every == 0:
                loss_avg = val_loss / print_every
                print("Validation : time = %dm, epoch %d, iter = %d, loss = %.10f, lr = %.10f, %ds per %d iters" % \
                      ((time.time() - start) // 60, epoch + 1, j + 1, loss_avg, get_lr(optimizer), time.time() - temp, print_every))
                val_loss = 0
                temp = time.time()
            j += 1
        val_steps = j + 1

        wandb.log({
            "epoch train loss": epoch_train_loss / train_steps,
            "epoch validation loss": epoch_val_loss / val_steps,
        })
        scheduler.step(epoch_val_loss / val_steps)

        epoch_train_loss = 0
        epoch_val_loss = 0

# Train from Checkpoint
def train_checkpoint(fpath, epochs, input_dir, output_dir, print_every=10):
    checkpoint = torch.load(fpath)
    model = model.load_state_dict(checkpoint)
    train(epochs, input_dir, output_dir, print_every)

# System Call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--checkpoint', type=bool)
    parser.add_argument('--checkpoint_fpath', type=str)
    args = parser.parse_args()

    if args.checkpoint == True:
        train_checkpoint(
            fpath=args.checkpoint_fpath,
            epochs=args.epoch, 
        )
    else:
        train(
            epochs=args.epoch, 
        )