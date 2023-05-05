# Source : https://tutorials.pytorch.kr/beginner/translation_transformer.html

# pip install -U torchdata
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence

import torch

# 원본 데이터의 링크가 동작하지 않으므로 데이터셋의 URL을 수정해야 합니다.
# 더 자세한 내용은 https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 을 참고해주세요.
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

class Tokenizer:
    def __init__(self, src_language, tgt_language):
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.token_transform = {}
        self.vocab_transform = {}

        # 출발어(source)와 목적어(target)의 토크나이저(tokenizer)를 생성합니다.
        self.token_transform[self.src_language] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[self.tgt_language] = get_tokenizer('spacy', language='en_core_web_sm')

        # 특수 기호(symbol)와 인덱스를 정의합니다
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        # 토큰들이 어휘집(vocab)에 인덱스 순서대로 잘 삽입되어 있는지 확인합니다
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tokenize()


    # 토큰 목록을 생성하기 위한 헬퍼(helper) 함수
    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        language_index = {self.src_language: 0, self.tgt_language: 1}
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])


    # Numerizer Dictionary
    def tokenize(self):
        for ln in [self.src_language, self.tgt_language]:
            # 학습용 데이터 반복자(iterator)
            train_iter = Multi30k(split='train', language_pair=(self.src_language, self.tgt_language))
            # torchtext의 Vocab(어휘집) 객체 생성
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 specials=self.special_symbols,
                                                                 special_first=True)
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)


    # 순차적인 작업들을 하나로 묶는 헬퍼 함수
    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func


    # BOS/EOS를 추가하고 입력 순서(sequence) 인덱스에 대한 텐서를 생성하는 함수
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS_IDX])))


    # 데이터를 텐서로 조합(collate)하는 함수
    def collate_fn(self, batch):
        # 출발어(src)와 도착어(tgt) 원시 문자열들을 텐서 인덱스로 변환하는 변형(transform)
        text_transform = {}
        for ln in [self.src_language, self.tgt_language]:
            text_transform[ln] = self.sequential_transforms(self.token_transform[ln], # 토큰화(Tokenization)
                                                            self.vocab_transform[ln], # 수치화(Numericalization)
                                                            self.tensor_transform) # BOS/EOS를 추가하고 텐서를 생성
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[self.src_language](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[self.tgt_language](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)

        return src_batch, tgt_batch


    # Source Language Vocabulary
    def src_vocab(self):
        return self.vocab_transform[self.src_language]


    # Target Language Vocabulary
    def tgt_vocab(self):
        return self.vocab_transform[self.tgt_language]
    
    # Convert Indices to words
    def idx_to_word(self, idx, vocab):
        words = []
        idx = idx[:30]
        for i in idx:
            word = vocab.get_itos()[i]
            if '<' not in word:
                words.append(word)
        words = " ".join(words)
        return words