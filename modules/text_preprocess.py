import pandas as pd
from torch.utils.data import DataLoader
from tokenizer_.tokenizer import Series_Tokenizer
from utils.utils import Make_vocab, PandasTextDataset
from sklearn.model_selection import train_test_split

class TextPreporcesser:

    def __init__(self, data_df, text_col, label_col,
                 padding_size, batch_size, shuffle=True,
                 tokenizer_mode='Mecab', split_data=True,
                 split_ratio=0.2, vocab_len_limit=None):

        self.data_df = data_df
        self.text_col = text_col
        self.label_col = label_col
        self.padding_size = padding_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer_mode = tokenizer_mode
        self.split_data = split_data
        self.split_ratio = split_ratio
        self.vocab_len_limit = vocab_len_limit

    def processing(self):
        self.tokenizer = Series_Tokenizer(mode=self.tokenizer_mode)
        # Make vocabulary
        corpus = self.data_df[self.text_col].values.tolist()
        make_vocab = Make_vocab(self.tokenizer, vocab_len_limit=self.vocab_len_limit)
        self.vocab = make_vocab(corpus)
        self.vocab_size = len(self.vocab)

        if self.split_data:
            # Set up Dataset
            x_train, x_test, y_train, y_test = train_test_split(self.data_df[self.text_col], self.data_df[self.label_col], test_size=0.2, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            train_set = PandasTextDataset(X=x_train, y=y_train, tokenizer=self.tokenizer, vocab=self.vocab, padding_size=self.padding_size)
            val_set = PandasTextDataset(X=x_val, y=y_val, tokenizer=self.tokenizer, vocab=self.vocab, padding_size=self.padding_size)
            test_set = PandasTextDataset(X=x_test, y=y_test, tokenizer=self.tokenizer, vocab=self.vocab, padding_size=self.padding_size)
            # Set dataloader
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=self.shuffle)
            self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=self.shuffle)
            self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=self.shuffle)
        else:
            dataset = PandasTextDataset(X=self.data_df[self.text_col], y=self.data_df[self.label_col], tokenizer=self.tokenizer, vocab=self.vocab, padding_size=self.padding_size)
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)




