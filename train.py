# QUAK Dataset : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=71268

from model import Transformer
from dataloader import dataloader
import torch.nn.functional as F

import torch
import torch.nn as nn
import time

d_model = 512
max_len = 256
n_heads = 8
N = 6
src_vocab = 50000
trg_vocab = 50000

model = Transformer(
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    d_model=512,
    max_len=max_len,
    n_heads=n_heads,
    N=N
)

# MPS
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Model initialization
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Adam Optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)

# Dataloader
train_iter = dataloader

# Training Step
def train(epochs, print_every=1):
    model.train()
    
    start = time.time()
    temp = start
    total_loss = 0
    
    for epoch in range(epochs):
        for i, value in enumerate(train_iter):
            src, trg = value
            source = torch.stack(src).transpose(-2, -1).to(device)
            target = torch.stack(trg).transpose(-2, -1).to(device)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            preds = model(source, target_input)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target_output, ignore_index=0)
            loss.backward()
            optim.step()
            
            total_loss += loss.data
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.10f, %ds per %d iters" % \
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, \
                       time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()