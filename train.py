# QUAK Dataset : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=71268

from model import Transformer

import torch
import torch.nn as nn

d_model = 512
max_len = 256
n_heads = 8
N = 6
src_vocab = 0
trg_vocab = 0

model = Transformer(
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    d_model=512,
    max_len=max_len,
    n_heads=n_heads,
    N=N
)

# Model initialization
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Adam Optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)