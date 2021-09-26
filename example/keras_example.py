from tensorflow.keras import layers, Input, Model, Sequential
import pandas as pd
import numpy as np
from utils.utils import Make_vocab, PandasTextDataset, pandas_split_dataDf
from tokenizer_.tokenizer import Series_Tokenizer
from sklearn.metrics import confusion_matrix

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
                                  drop_not_in_vocab = True,
                                  mode=None)

val_dataset = PandasTextDataset(text=val_df['comment'],
                                  label=val_df['label'],
                                  tokenizer=tokenizer,
                                  vocab=vocab,
                                  padding_size=padding_size,
                                  drop_not_in_vocab=True,
                                mode=None)

test_dataset = PandasTextDataset(text=test_df['comment'],
                                 label=test_df['label'],
                                 tokenizer=tokenizer,
                                 vocab=vocab,
                                 padding_size=padding_size,
                                 drop_not_in_vocab=True,
                                 mode=None)


# KERAS
from tensorflow.keras import losses, optimizers, metrics
import tensorflow as tf
# Modeling using LSTM or GRU
from keras.layers import Embedding, Dense, LSTM, GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def numpy_one_hot_from_dataset(dataset, vocab_len, num_classes):

    x_array_appender = []
    y_appender = []
    for j, (x, y) in enumerate(dataset):

        x = x.numpy()
        y = y.numpy()
        arrayX = np.zeros((len(x), vocab_len))
        for i, idx in enumerate(x):
            if idx != 0 :
                arrayX[i, idx] = 1
        arrayY = np.zeros(num_classes)
        arrayY[y] = 1
        x_array_appender.append(arrayX)
        y_appender.append(arrayY)
        if j == len(dataset)-1:
            break

    return np.array(x_array_appender), np.array(y_appender)


def one_hot_encoder(array1d, col_dim):
    array = np.zeros((len(array1d), col_dim))
    for i, idx in enumerate(array1d):
        if idx != 0:
            array[i, idx] = 1
    return array


def dataset_to_array(dataset):
    X = []
    Y = []
    for i, (x, y) in enumerate(dataset):
        X.append(x)
        Y.append(y)
        if i == len(dataset)-1:
            break
    return np.array(X), np.array(Y)


def make_model(padding_size, vocab_len, hidden_size, output_size):
    input = Input(shape=(padding_size, ))
    emb_layer = layers.Embedding(vocab_len, hidden_size, input_length=padding_size)
    lstm_layer = layers.LSTM(hidden_size)
    fc_layer = layers.Dense(output_size, activation='softmax')

    # forward
    x0 = emb_layer(input)
    x1 = lstm_layer(x0)
    output = fc_layer(x1)

    return Model(input, output)

def sequence_model(vocab_len, hidden_size, output_size):
    model = Sequential()
    model.add(Embedding(vocab_len, hidden_size))
    model.add(LSTM(hidden_size))  # LSTM or GRU
    model.add(Dense(output_size, activation='sigmoid'))
    return model

def get_accuracy(pred, label):
    pred = pred.squeeze()
    label = label.squeeze()
    return np.sum([1 for x, y in zip(pred, label) if int(x) == int(y)]) / len(pred)

# data
x_train, y_train = dataset_to_array(train_dataset)
x_val, y_val = dataset_to_array(val_dataset)
x_test, y_test = dataset_to_array(test_dataset)

# complie
i_model = make_model(padding_size=30, vocab_len=len(vocab), hidden_size=64, output_size=1)
i_model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.BinaryCrossentropy(), metrics=metrics.Accuracy())

s_model = sequence_model(vocab_len=len(vocab), hidden_size=64, output_size=1)
s_model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.BinaryCrossentropy(), metrics=metrics.Accuracy())

# model fit
i_model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32)
s_model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# check
model_select = '1'

if model_select == 'model':
    pred = i_model.predict(x_train)
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    print(confusion_matrix(pred, y_train))

    pred = i_model.predict(x_val)
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    print(confusion_matrix(pred, y_val))

    pred = i_model.predict(x_test)
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    print(confusion_matrix(pred, y_test))
else:
    pred = s_model.predict(x_train)
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    print(confusion_matrix(pred, y_train))

    pred = s_model.predict(x_val)
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    print(confusion_matrix(pred, y_val))

    pred = s_model.predict(x_test)
    pred = list(map(lambda x: 1 if x > 0.5 else 0, pred))
    print(confusion_matrix(pred, y_test))



