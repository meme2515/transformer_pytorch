from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import torch
import pandas as pd

class QUAK(Dataset):
    def __init__(self, eng_file, kor_file):
        self.eng_feed = [line.strip() for line in open(eng_file).readlines()]
        self.kor_feed = [line.strip() for line in open(kor_file).readlines()]

    def __len__(self):
        return len(self.eng_feed)

    def __getitem__(self, idx):
        return self.eng_feed[idx], self.kor_feed[idx]
    
dataset = QUAK("overfit_test/h_based_16m.pe", "overfit_test/h_based_16m.src")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
