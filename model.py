# Dimensions :  https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
# Implementation #1 : https://www.youtube.com/watch?v=U0s0f995w14
# Implementation #2 : https://cpm0722.github.io/pytorch-implementation/transformer
# Implementation #3 : https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Scaled Dot-Product Attention
def self_attention(q, k, v, mask=None):
    # query, key, value: (n_batch, h, seq_len, d_k)
    # mask: (n_batch, 1, seq_len, seq_len)
    d_k = k.shape[-1]
    scores = torch.matmul(q, torch.transpose(k, -2, -1)) # (n_batch, h, seq_len, seq_len)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask==0, -1e9)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v) # (n_batch, h, seq_len, d_k)
    return output

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert (self.d_k * n_heads == d_model), "Model size (embedding length) needs to be divisible by heads."

        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, values, keys, queries, mask=None):
        # query, key, value : (n_batch, seq_len, d_model)
        # mask : (n_batch, seq_len, seq_len)
        # returns : (n_batch, n_heads, seq_len, d_k)
        n_batch = queries.shape[0]

        queries = self.q_linear(queries).view(n_batch, -1, self.n_heads, self.d_k) # (n_batch, seq_len, n_heads, d_k)
        keys    = self.k_linear(keys).view(n_batch, -1, self.n_heads, self.d_k)    # (n_batch, seq_len, n_heads, d_k)
        values  = self.v_linear(values).view(n_batch, -1, self.n_heads, self.d_k)  # (n_batch, seq_len, n_heads, d_k)

        queries = queries.transpose(1,2) # (n_batch, n_heads, seq_len, d_k)
        keys    = keys.transpose(1,2)    # (n_batch, n_heads, seq_len, d_k)
        values  = values.transpose(1,2)  # (n_batch, n_heads, seq_len, d_k)

        scores = self_attention(queries, keys, values, mask=mask) # (n_batch, n_heads, seq_len, d_k)
        scores = scores.transpose(1,2) # (n_batch, seq_len, n_heads, d_k)
        concat = scores.contiguous().view(n_batch, -1, self.d_model)
        output = self.out_linear(concat)

        return output