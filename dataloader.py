# Tokenizer : https://www.analyticsvidhya.com/blog/2021/09/an-explanatory-guide-to-bert-tokenizer/

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np

class QUAK(Dataset):
    def __init__(self, eng_file, kor_file):
        self.eng_feed = [line.strip() for line in open(eng_file).readlines()]
        self.kor_feed = [line.strip() for line in open(kor_file).readlines()]

    def __len__(self):
        return len(self.eng_feed)

    def __getitem__(self, idx):
        eng_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        kor_tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

        eng_tokens = eng_tokenizer.encode(self.eng_feed[idx], padding="max_length", max_length=256, truncation=True)
        kor_tokens = kor_tokenizer.encode(self.kor_feed[idx], padding="max_length", max_length=256, truncation=True)

        return eng_tokens, kor_tokens
    
dataset = QUAK("overfit_test/h_based_16m.pe", "overfit_test/h_based_16m.src")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)