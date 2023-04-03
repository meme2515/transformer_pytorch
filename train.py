# QUAK Dataset : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=71268

from model.model import Transformer
from model.dataloader import make_generator
from torch.optim.lr_scheduler import StepLR

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
SRC_VOCAB = 50000
TRG_VOCAB = 50000
LEARNING_RATE = 0.0001
BETA1, BETA2 = 0.9, 0.98

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
    trg_vocab=TRG_VOCAB,
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
optim = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE, 
    betas=(BETA1, BETA2), 
    eps=1e-9
)

# LR Scheduler
scheduler = StepLR(optim, step_size=10, gamma=0.1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Training Step
def train(epochs, input_dir, output_dir, print_every=10):
    model.train()
    
    start = time.time()
    temp = start
    train_loss = 0
    val_loss = 0
    epoch_train_loss = 0
    epoch_val_loss = 0
    train_steps = 0
    val_steps = 0

    train_dataloader, val_dataloader, _ = make_generator(input_dir, output_dir)
    
    for epoch in range(epochs):
        # Train Step
        for i, value in enumerate(train_dataloader):
            src, trg = value
            source = torch.stack(src).transpose(-2, -1).to(DEVICE)
            target = torch.stack(trg).transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            model.train(True)
            preds = model(source, target_input)
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=0)
            loss.backward()
            optim.step()
            wandb.log({"step train loss": loss})

            train_loss += loss.data
            epoch_train_loss += loss.data

            if (i + 1) % print_every == 0:
                loss_avg = train_loss / print_every
                print("Train : time = %dm, epoch %d, iter = %d, loss = %.10f, lr = %.10f, %ds per %d iters" % \
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, get_lr(optim), time.time() - temp, print_every))
                train_loss = 0
                temp = time.time()
                
        scheduler.step()
        train_steps = i + 1

        if (epoch + 1) % 10 == 0:
            torch.save(model, 'checkpoint_ep_%d.pt'%(epoch + 1))
        
        # Validation Step
        for j, value in enumerate(val_dataloader):
            src, trg = value
            source = torch.stack(src).transpose(-2, -1).to(DEVICE)
            target = torch.stack(trg).transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            model.train(False)
            preds = model(source, target_input)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=0)

            epoch_val_loss += loss.data

            if (j + 1) % print_every == 0:
                loss_avg = val_loss / print_every
                print("Validation : time = %dm, epoch %d, iter = %d, loss = %.10f, lr = %.10f, %ds per %d iters" % \
                      ((time.time() - start) // 60, epoch + 1, j + 1, loss_avg, get_lr(optim), time.time() - temp, print_every))
                val_loss = 0
                temp = time.time()
        val_steps = j + 1

        wandb.log({
            "epoch train loss": epoch_train_loss / train_steps,
            "epoch validation loss": epoch_val_loss / val_steps,
        })

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
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=bool)
    parser.add_argument('--checkpoint_fpath', type=str)
    args = parser.parse_args()

    if args.checkpoint == True:
        train_checkpoint(
            fpath=args.checkpoint_fpath,
            epochs=args.epoch, 
            input_dir=args.input_dir, 
            output_dir=args.output_dir,
        )
    else:
        train(
            epochs=args.epoch, 
            input_dir=args.input_dir, 
            output_dir=args.output_dir,
        )