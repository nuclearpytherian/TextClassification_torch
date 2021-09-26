from itertools import chain
import re
from collections import Counter
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

class Make_vocab:
    """
    To make vocabulary using corpus
    """

    def __init__(self, tokenizer, vocab_len_limit=None, character_only=True):
        self.tokenizer = tokenizer
        self.vocab_len_limit = vocab_len_limit
        self.character_only = character_only

    def __call__(self, corpus:list):
        # To remove characters except korean, english, and number
        if self.character_only == True:
            tokend_word = list(chain(*[self.tokenizer(self.remove_not_character(s)) for s in corpus]))

        elif self.character_only == False:
            tokend_word = list(chain(*[self.tokenizer(s) for s in corpus]))
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


class PandasTextDataset(Dataset):

    def __init__(self, text:"pandas.core.series.Series", label:"pandas.core.series.Series", tokenizer, vocab, padding_size, drop_not_in_vocab=True, mode='torch'):
        super(PandasTextDataset, self).__init__()
        self.x = text
        self.y = label
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.padding_size = padding_size
        self.drop_not_in_vocab = drop_not_in_vocab
        self.mode = mode

    def __getitem__(self, index):
        x_data = self.x[index]
        y_data = self.y[index]
        x_tokend = self.tokenizer(x_data)
        x_idx = self.padding(self.token2index(x_tokend))
        if self.mode == 'torch':
            x_return = torch.Tensor(x_idx).int()
            y_return = torch.Tensor([y_data]).long().squeeze()
        else:
            x_return = np.array(x_idx)
            y_return = np.array([y_data])
        return x_return, y_return

    def __len__(self):
        return len(self.x)

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


class SimplePredictor:
    def __init__(self, model_origin, artifact_saved_path, tokenizer, vocab, padding_size, drop_not_in_vocab=True):
        self.model_origin = model_origin
        self.artifact_saved_path = artifact_saved_path
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.padding_size = padding_size
        self.drop_not_in_vocab = drop_not_in_vocab

    def __call__(self, sentence):
        data = self.return_input(sentence)
        model = self.load_model()
        return model(data)

    def return_input(self, sentence):
        data = self.padding(self.token2index(self.tokenizer(sentence)))
        return torch.Tensor(data).int().view(1,-1)

    def load_model(self):
        model = self.model_origin
        model.load_state_dict(torch.load(self.artifact_saved_path))
        return model.eval()

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


def pandas_split_dataDf(data_df, test_size = 0.2, val_size = 0.2, random_state=42):
    random.seed(random_state)
    original_idx = list(range(len(data_df)))
    test_idx = random.sample(original_idx, int(len(original_idx) * test_size))
    train_idx = [i for i in original_idx if i not in test_idx]
    val_idx = random.sample(train_idx, int(len(train_idx) * val_size))
    train_idx = [i for i in train_idx if i not in val_idx]
    return data_df.iloc[train_idx].reset_index(drop=True), data_df.iloc[val_idx].reset_index(drop=True), data_df.iloc[test_idx].reset_index(drop=True)


def split_dataset(dataset, test_size = 0.2, val_size = 0.2, random_state=42):
    random.seed(random_state)
    original_idx = list(range(len(dataset)))
    test_idx = random.sample(original_idx, int(len(original_idx) * test_size))
    train_idx = [i for i in original_idx if i not in test_idx]
    val_idx = random.sample(train_idx, int(len(train_idx) * val_size))
    train_idx = [i for i in train_idx if i not in val_idx]
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


def remove_not_character(string):
    # To remove characters except korean, english, and number
    pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+'
    return re.sub(pattern=pattern, repl='', string=string)



class Predictor:
    def __init__(self, trained_model, dataloader):
        self.model = trained_model
        self.dataloader = dataloader

    def __call__(self, dataloader):
        preds = []
        for i, (x, y) in enumerate(dataloader):
            pred = self.model_predict(self.model, x)
            preds.append(int(pred))
            if i == len(dataloader)-1:
                break
        return preds

    def model_predict(self, model, input):
        if input.ndim == 1:
            input = input.unsqueeze(0)
        pred = model(input).data.max(1, keepdim=True)[1].squeeze()
        return pred

    def get_preds(self):
        preds = []
        labels = []
        for i, (x, y) in enumerate(self.dataloader):
            pred = self.model_predict(self.model, x)
            preds.append(int(pred))
            labels.append(int(y))
            if i == len(self.dataloader) - 1:
                break
        return preds, labels

    def get_accuracy(self):
        preds, labels = self.get_preds()
        return np.sum([1 for x, y in zip(preds, labels) if int(x) == int(y)]) / len(preds)

    def confusion_matrix(self):
        preds, labels = self.get_preds()
        df = pd.DataFrame(confusion_matrix(preds, labels))
        return df

