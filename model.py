# Dimensions :  https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
# Implementation #1 : https://www.youtube.com/watch?v=U0s0f995w14
# Implementation #2 : https://cpm0722.github.io/pytorch-implementation/transformer
# Implementation #3 : https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

# Masked Self-Attention : https://www.youtube.com/watch?v=piT1_k8b9uM

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_clones(module, N):
    return nn.ModuleList([module.copy.deepcopy() for i in range(N)])

# Initial Word Embedding
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        return self.embed(input)
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_embed=512, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()

        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.encoding = encoding.unsqueeze(0)

    def forward(self, input):
        seq_len = input.size()[1]
        pos_embed = self.encoding[:, :seq_len, :]
        out = input + pos_embed

        return out

# Scaled Dot-Product Attention
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, queries, keys, values, mask=None):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, 1, seq_len, seq_len)
        d_k = keys.shape[-1]
        scores = torch.matmul(queries, torch.transpose(keys, -2, -1)) # (n_batch, h, seq_len, seq_len)
        scores = scores / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask==0, -1e9)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, values) # (n_batch, h, seq_len, d_k)
        return output

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert (self.d_k * n_heads == d_model), "Model size (embedding length) needs to be divisible by heads."

        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.d_model)
        self.self_attention = SelfAttention()

    def forward(self, values, keys, queries, mask=None):
        # query, key, value : (n_batch, seq_len, d_model)
        # mask : (n_batch, seq_len, seq_len)
        # returns : (n_batch, n_heads, d_model)
        n_batch = queries.shape[0]

        queries = self.q_linear(queries).view(n_batch, -1, self.n_heads, self.d_k) # (n_batch, seq_len, n_heads, d_k)
        keys    = self.k_linear(keys).view(n_batch, -1, self.n_heads, self.d_k)
        values  = self.v_linear(values).view(n_batch, -1, self.n_heads, self.d_k)

        queries = queries.transpose(1,2) # (n_batch, n_heads, seq_len, d_k)
        keys    = keys.transpose(1,2)
        values  = values.transpose(1,2)

        scores = self.self_attention(queries, keys, values, mask=mask) # (n_batch, n_heads, seq_len, d_k)
        scores = scores.transpose(1,2) # (n_batch, seq_len, n_heads, d_k)
        concat = scores.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        output = self.out_linear(concat) # (n_batch, seq_len, d_model)

        return output

# Feed Forward Net
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff) 
        self.fc2 = nn.Linear(d_ff, d_model) 

    def forward(self, input):
        output = F.relu(self.fc1(input)) # (n_batch, seq_len, d_ff)
        output = self.fc2(output) # (n_batch, seq_len, d_model)

        return output

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, d_model=512, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, input):
        norm = self.alpha * (input - input.mean(dim=-1, keepdim=True)) \
            / (input.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(EncoderBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
    
    def forward(self, input):
        # input : (n_batch, seq_len, d_model)
        # final output : (n_batch, seq_len, d_model)
        output = self.multihead_attention(input, input, input) # (n_batch, seq_len, d_model)
        output = input + output
        output = self.layernorm1(output)

        output2 = self.feedforward(output) # (n_batch, seq_len, d_model)
        output2 = output + output2
        output2 = self.layernorm2(output2)

        return output2
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads)
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.layernorm3 = LayerNorm(d_model)

    def forward(self, input, e_output, mask):
        output = self.masked_attention(input, input, input, mask) # (n_batch, seq_len, d_model)
        output = input + output
        output = self.layernorm1(output)

        # "the queries come from the previous decoder layer, 
        # and the memory keys and values come from the output of the encoder.""
        output2 = self.multihead_attention(e_output, e_output, output) # (n_batch, seq_len, d_model)
        output2 = output + output2
        output2 = self.layernorm2(output2)

        output3 = self.feedforward(output2) # (n_batch, seq_len, d_model)
        output3 = output2 + output3
        output3 = self.layernorm3(output3)

        return output3
    
# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self):
        return

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self):
        return

# Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def forward(self):
        return
    
# BERT
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()

    def forward(self):
        return

# GPT
class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()

    def forward(self):
        return