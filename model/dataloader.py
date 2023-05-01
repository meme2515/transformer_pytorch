# # Tokenizer : https://tutorials.pytorch.kr/beginner/translation_transformer.html

# from torch.utils.data import Dataset, DataLoader, random_split
# from transformers import BertTokenizerFast

# import torch.nn.functional as F
# import torch
# import pandas as pd
# import numpy as np

# SRC_LANGUAGE = 'de'
# TGT_LANGUAGE = 'en'

# class QUAK(Dataset):
#     def __init__(self, eng_file, kor_file):
#         self.eng_feed = [line.strip() for line in open(eng_file).readlines()]
#         self.kor_feed = [line.strip() for line in open(kor_file).readlines()]

#     def __len__(self):
#         return len(self.eng_feed)

#     def __getitem__(self, idx):
#         eng_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
#         kor_tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

#         eng_tokens = eng_tokenizer.encode(self.eng_feed[idx], padding="max_length", max_length=64, truncation=True)
#         kor_tokens = kor_tokenizer.encode(self.kor_feed[idx], padding="max_length", max_length=64, truncation=True)

#         return eng_tokens, kor_tokens

# def make_generator(input_dir, output_dir):
#     dataset = QUAK(input_dir, output_dir)
#     generator = torch.Generator().manual_seed(42)
#     train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

#     train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
#     val_dataloader = DataLoader(val_set, batch_size=16, shuffle=True)
#     test_dataloader = DataLoader(test_set, batch_size=16, shuffle=True)

#     return train_dataloader, val_dataloader, test_dataloader