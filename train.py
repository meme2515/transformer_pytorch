# QUAK Dataset : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=71268

from model import Transformer
from dataloader import train_dataloader, val_dataloader, test_dataloader
import torch.nn.functional as F

import torch
import torch.nn as nn
import time
import wandb
import os

# conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 512
MAX_LEN = 256
N_HEADS = 8
N = 6
SRC_VOCAB = 50000
TRG_VOCAB = 50000
LEARNING_RATE = 0.0001
EPOCHS = 1
BETA1, BETA2 = 0.9, 0.98

wandb.init(
    project="transformer v1",

    config={
        "learning_rate" : LEARNING_RATE,
        "architecture" : "Transformer",
        "dataset" : "QUAK-H",
        "epochs" : EPOCHS,
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

# Training Step
def train(epochs, print_every=100):
    model.train()
    
    start = time.time()
    temp = start
    total_train_loss = 0
    total_val_loss = 0
    
    for epoch in range(epochs):
        # Train Step
        for i, value in enumerate(train_dataloader):
            src, trg = value
            source = torch.stack(src).transpose(-2, -1).to(DEVICE)
            target = torch.stack(trg).transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            preds = model(source, target_input)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=0)
            loss.backward()
            optim.step()
            wandb.log({"train loss": loss})


            total_train_loss += loss.data
            if (i + 1) % print_every == 0:
                loss_avg = total_train_loss / print_every
                print("Train : time = %dm, epoch %d, iter = %d, loss = %.10f, %ds per %d iters" % \
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, \
                       time.time() - temp, print_every))
                total_train_loss = 0
                temp = time.time()

        # Validation Step
        # for i, value in enumerate(val_dataloader):
        #     src, trg = value
        #     source = torch.stack(src).transpose(-2, -1).to(DEVICE)
        #     target = torch.stack(trg).transpose(-2, -1).to(DEVICE)

        #     target_input = target[:, :-1]
        #     target_output = target[:, 1:].contiguous().view(-1)

        #     preds = model(source, target_input)
        #     loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=0)

        #     wandb.log({"validation loss": loss})
        #     total_val_loss += loss.data
        #     if (i + 1) % print_every == 0:
        #         loss_avg = total_val_loss / print_every
        #         print("Validation : time = %dm, epoch %d, iter = %d, loss = %.10f, %ds per %d iters" % \
        #               ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, \
        #                time.time() - temp, print_every))
        #         total_val_loss = 0
        #         temp = time.time()

if __name__ == "__main__":
    train(EPOCHS)
    torch.save(model.state_dict(), 'model.pt')