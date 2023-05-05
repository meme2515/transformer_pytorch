# :fire: PyTorch Transformer

This is my implementation of the widely known Transformer architecture from the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Training was done on the [Multi30k dataset](https://github.com/multi30k/dataset), which provides 31,014 instances of images and corresponding German-English descriptions. Although the intention is to reproduce the original paper's implementation, this version may include minor differences. Please feel free to raise issues if you find that improvements could be made in any way.

## The Transformer Architecture

```
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, max_len=256, n_heads=8, d_ff=2048, N=6, drop_prob=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, d_model, max_len, n_heads, d_ff, N, drop_prob)
        self.decoder = Decoder(tgt_vocab, d_model, max_len, n_heads, d_ff, N, drop_prob)
        self.linear = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        # src : (n_batch, src_seq_len)
        # tgt : (n_batch, tgt_seq_len)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)

        e_output = self.encoder(src, src_mask) # (n_batch, src_seq_len, d_model)
        d_output = self.decoder(tgt, e_output, tgt_mask, src_tgt_mask) # (n_batch, tgt_seq_len, d_model)
        output = self.linear(d_output) # (n_batch, tgt_seq_len, tgt_vocab)

        return output
```

### Self Attention

### Multihead Attention

### The Encoder Module

### The Decoder Module

### Masking

## Train & Validation Loss

<img src = "images/loss.png" width="600">