from model import model
from model.dataloader import make_generator
import torch
import numpy as np

input_dir = "QUAK-H/h_based_16m.pe"
output_dir = "QUAK-H/h_based_16m.src"

train_dataloader, val_dataloader, _ = make_generator(input_dir, output_dir)
src, trg = next(iter(train_dataloader))

source = torch.stack(src).transpose(-2, -1)
target = torch.stack(trg).transpose(-2, -1)

print(source[0])

transformer = model.Transformer(100, 100)
src_mask_result = transformer.make_src_mask(source)
print(src_mask_result[0][0][0])