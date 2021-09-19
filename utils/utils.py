from itertools import chain
import re
from collections import Counter
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd

class Make_vocab:
    """
    To make vocabulary using corpus
    """

    def __init__(self, tokenizer, vocab_len_limit=None, character_only=True):
        self.tokenizer = tokenizer
        self.vocab_len_limit = vocab_len_limit
        self.character_only = character_only

    def __call__(self, copus:list):
        # To remove characters except korean, english, and number
        if self.character_only == True:
            tokend_word = list(chain(*[self.tokenizer(self.remove_not_character(s)) for s in copus]))

        elif self.character_only == False:
            tokend_word = list(chain(*[self.tokenizer(s) for s in copus]))
        # To remove low frequent words
        if self.vocab_len_limit is None:
            word_set = set(tokend_word)
        else:
            word_set = set([s for s, i in Counter(tokend_word).most_common(self.vocab_len_limit)])

        vocab = {word: i + 2 for i, word in enumerate(word_set)}
        vocab['<unk>'] = 0
        vocab['<pad>'] = 1
        return vocab

    def remove_not_character(self, string):
        # To remove characters except korean, english, and number
        pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+'
        return re.sub(pattern=pattern, repl='', string=string)


class DFDataset(Dataset):

    def __init__(self, X, y, tokenizer, vocab, padding_size, drop_not_in_vocab=True):
        super(DFDataset, self).__init__()

        if type(X) == np.ndarray:
            self.x = X
        elif type(X) == pd.core.series.Series:
            self.x = X.values

        if type(y) == np.ndarray:
            self.y = y
        elif type(y) == pd.core.series.Series:
            self.y = y.values

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.padding_size = padding_size
        self.drop_not_in_vocab = drop_not_in_vocab

    def __getitem__(self, index):
        x_data = self.x
        y_data = self.y
        x_tokend = [self.tokenizer(s) for s in x_data]
        x_idx =[self.padding(self.token2index(s)) for s in x_tokend]
        x_return = torch.Tensor(x_idx).int()
        y_return = torch.Tensor(y_data).int()
        return x_return[index], y_return[index]

    def __len__(self):
        return len(self.y)

    def token2index(self, tokend):
        idxes = []
        for t in tokend:
            try:
                idxes.append(self.vocab[t])
            except KeyError:
                idxes.append(self.vocab['<unk>'])
        if self.drop_not_in_vocab:
            idxes = [t for t in idxes if t != 0]

        return idxes

    def padding(self, idxes):

        if len(idxes) > self.padding_size:
            idxes = idxes[:self.padding_size]

        elif len(idxes) < self.padding_size:
            idxes = idxes + [0 for x in range(self.padding_size - len(idxes))]

        return idxes


def split_dataset(dataset, test_size = 0.15, val_size = 0.2, random_state=42):
    random.seed(random_state)
    original_idx = list(range(len(dataset)))
    test_idx = random.sample(original_idx, int(len(original_idx) * test_size))
    train_idx = [i for i in original_idx if i not in test_idx]
    val_idx = random.sample(train_idx, int(len(train_idx) * val_size))
    train_idx = [i for i in train_idx if i not in val_idx]
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]

