from model.model import Transformer
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from util import tokenizer

import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import wandb
import argparse

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 512
MAX_LEN = 256
N_HEADS = 8
N = 6
LEARNING_RATE = 0.0001
BETA1, BETA2 = 0.9, 0.98
WARMUP = 30
CLIP = 1
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Count Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Weight Initializer
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

# Current Learning Rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
tokenizer = tokenizer.Tokenizer(SRC_LANGUAGE, TGT_LANGUAGE)
src_vocab = tokenizer.src_vocab()
tgt_vocab = tokenizer.tgt_vocab()

wandb.init(
    project="transformer v1",
)

# Transformer
model = Transformer(
    src_vocab=len(src_vocab),
    tgt_vocab=len(tgt_vocab),
    d_model=D_MODEL,
    max_len=MAX_LEN,
    n_heads=N_HEADS,
    N=N,
)

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
                                                       factor=0.5, 
                                                       patience=3)

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# MPS & CUDA
model.to(DEVICE)
model.apply(initialize_weights)

wandb.watch(model, log=None)

# Training Step
def train(model, criterion, dataloader, optimizer, clip):
    model.train()
    train_loss, train_steps = 0, 0

    # Train Step
    for src, tgt in dataloader:
        source = src.transpose(-2, -1).to(DEVICE)
        target = tgt.transpose(-2, -1).to(DEVICE)

        target_input = target[:, :-1]
        target_output = target[:, 1:].contiguous().view(-1)

        optimizer.zero_grad()
        preds = model(source, target_input)
            
        loss = criterion(preds.view(-1, preds.size(-1)), target_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        wandb.log({"step train loss": loss})

        train_loss += loss.data
        train_steps += 1

    return train_loss / (train_steps + 1)

# Evaluation Step
def evaluate(model, criterion, dataloader, tokenizer, vocab):
    model.eval()
    val_loss, val_steps = 0, 0

    with torch.no_grad():
        for src, tgt in dataloader:
            source = src.transpose(-2, -1).to(DEVICE)
            target = tgt.transpose(-2, -1).to(DEVICE)

            target_input = target[:, :-1]
            target_output = target[:, 1:].contiguous().view(-1)

            preds = model(source, target_input)
            loss = criterion(preds.view(-1, preds.size(-1)), target_output)
            val_loss += loss.data
            val_steps += 1

        predicted_sentence = torch.argmax(preds.view(-1, preds.size(-1)), dim=1)
        predicted_sentence = tokenizer.idx_to_word(predicted_sentence, vocab)
        original_sentence = tokenizer.idx_to_word(target_output, vocab)
        print("Predicted Sentence :", predicted_sentence)
        print("Original Sentence :", original_sentence)
    
    return val_loss / (val_steps + 1)

# Train & Evaluation Run
def run(epochs, best_loss):
    print(f'The model has {count_parameters(model):,} trainable parameters')

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=128, collate_fn=tokenizer.collate_fn, shuffle=True)
    eval_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    eval_dataloader = DataLoader(eval_iter, batch_size=128, collate_fn=tokenizer.collate_fn, shuffle=True)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, criterion, train_dataloader, optimizer, CLIP)
        eval_loss = evaluate(model, criterion, eval_dataloader, tokenizer, tgt_vocab)
        end_time = time.time()

        if epoch > WARMUP:
            scheduler.step(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), 'model.pt')
        
        wandb.log({
            "epoch train loss": train_loss,
            "epoch validation loss": eval_loss,
        })

        print("Epoch %d : time = %dm, train_loss = %.10f, eval_loss = %.10f, lr = %.10f" % \
              (epoch + 1, (end_time - start_time) // 60, train_loss, eval_loss, get_lr(optimizer)))
        

# System Call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    run(epochs=args.epochs, best_loss=100)