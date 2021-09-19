import torch
from torch.utils.data import DataLoader
from tokenizer_.tokenizer import Series_Tokenizer
from utils.utils import Make_vocab, DFDataset, split_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__=="__main__":
    data = pd.read_csv('data/movie_naver.csv')
    data = data.dropna()

    # Set tokenizer : Mecab
    tokenizer = Series_Tokenizer(mode='Mecab')

    # Make vocabulary
    copus = data['comment'].values.tolist()
    make_vocab = Make_vocab(tokenizer, vocab_len_limit=10000)
    vocab = make_vocab(copus)

    # Set up Dataset
    x_train, x_test, y_train, y_test = train_test_split(data['comment'], data['label'], test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    padding_size = 30
    train_set = DFDataset(X=x_train, y=y_train, tokenizer=tokenizer, vocab=vocab, padding_size=padding_size)
    val_set = DFDataset(X=x_val, y=y_val, tokenizer=tokenizer, vocab=vocab, padding_size=padding_size)
    test_set = DFDataset(X=x_test, y=y_test, tokenizer=tokenizer, vocab=vocab, padding_size=padding_size)

    # Set dataloader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Check using next/iter
    train_features, train_labels = next(iter(train_loader))

    # Model
    from modules.model import TextClassifierLSTM

    model = TextClassifierLSTM(VOCAB_SIZE=len(vocab), EMB_SIZE=64, HIDDEN_SIZE=64, NUM_CLASSES=1, N_LAYER=1)
    model.train()
    print(model(train_features))






