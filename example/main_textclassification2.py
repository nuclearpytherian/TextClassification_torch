
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.utils import Make_vocab, PandasTextDataset, pandas_split_dataDf, Predictor
from tokenizer_.tokenizer import Series_Tokenizer
from modules.model import TextClassificationModel
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
    padding_size = 30
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

    # model
    VOCAB_SIZE = len(vocab)
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 2
    model = TextClassificationModel(vocab_size=VOCAB_SIZE,
                               embed_dim=HIDDEN_SIZE,
                               num_class=OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, eps=1e-8)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    early_stopping = EarlyStopping(patience=10, verbose=False)

    # Train
    Training = TextTrainer(train_dataloader,
                           val_dataloader,
                           model,
                           criterion,
                           optimizer,
                           scheduler=scheduler,
                           early_stoper=early_stopping,
                           EPOCH=100)
    Training.train()

    # Plot loss graph
    Training.plot_loss_graph()

    # Save model
    best_model = True
    Training.save_model(best_model=best_model)

    # Eval
    load_model = TextClassificationModel(vocab_size=VOCAB_SIZE,
                                    embed_dim=HIDDEN_SIZE,
                                    num_class=OUTPUT_SIZE)
    load_model.load_state_dict(torch.load('artifact/best_epoch_model.pt' if best_model == True else 'artifact/last_epoch_model.pt'))
    load_model.eval()

    predictor = Predictor(load_model, train_dataset)
    acc = predictor.confusion_matrix()
    print(acc)

    predictor = Predictor(load_model, val_dataset)
    acc = predictor.confusion_matrix()
    print(acc)

    predictor = Predictor(load_model, test_dataset)
    acc = predictor.confusion_matrix()
    print(acc)
