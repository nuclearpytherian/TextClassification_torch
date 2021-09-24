
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.utils import Make_vocab, PandasTextDataset, pandas_split_dataDf
from tokenizer_.tokenizer import Series_Tokenizer
from modules.model import TextLSTMClassifier
from modules.train import TextTrainer
from modules.earlystop import EarlyStopping
import pandas as pd

if __name__ == "__main__":
    # data
    data_df = pd.read_csv('data/movie_naver.csv')
    data_df = data_df.dropna()

    # split dataset
    train_df, val_df, test_df = pandas_split_dataDf(data_df, test_size=0.2, val_size=0.2)

    # tokenizer & vocab
    tokenizer = Series_Tokenizer(mode='mecab')
    corpus = data_df['comment'].values
    vocab_generator = Make_vocab(tokenizer, vocab_len_limit=None, character_only=True)
    vocab = vocab_generator(corpus)

    # dataset
    padding_size=30
    train_dataset = PandasTextDataset(text=train_df['comment'],
                                      label=train_df['label'],
                                      tokenizer=tokenizer,
                                      vocab=vocab,
                                      padding_size=padding_size,
                                      drop_not_in_vocab = True)

    val_dataset = PandasTextDataset(text=val_df['comment'],
                                      label=val_df['label'],
                                      tokenizer=tokenizer,
                                      vocab=vocab,
                                      padding_size=padding_size,
                                      drop_not_in_vocab=True)

    test_dataset = PandasTextDataset(text=test_df['comment'],
                                     label=test_df['label'],
                                     tokenizer=tokenizer,
                                     vocab=vocab,
                                     padding_size=padding_size,
                                     drop_not_in_vocab=True)

    # dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=32,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=32,
                                  shuffle=True)

    # model
    model = TextLSTMClassifier(VOCAB_SIZE=len(vocab),
                               EMB_SIZE=64,
                               HIDDEN_SIZE=64,
                               OUTPUT_SIZE=2,
                               N_LAYER=1,
                               DROPOUT=0.0,
                               bidirectional = False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=10, verbose=False)

    # Train
    Training = TextTrainer(train_dataloader,
                           val_dataloader,
                           model,
                           criterion,
                           optimizer,
                           scheduler=None,
                           early_stoper=early_stopping,
                           EPOCH=20)
    Training.train()

    # Save model
    Training.save_model(best_model=True)

    # Plot loss graph
    Training.plot_loss_graph()