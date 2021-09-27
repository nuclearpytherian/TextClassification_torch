import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class TextLSTMClassifier(nn.Module):

    def __init__(self, VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYER, DROPOUT, bidirectional=False):
        super(TextLSTMClassifier, self).__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMB_SIZE = EMB_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.N_LAYER = N_LAYER
        self.DROPOUT = DROPOUT
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.EMB_SIZE)
        self.lstm = nn.LSTM(self.EMB_SIZE, self.HIDDEN_SIZE, num_layers=self.N_LAYER, batch_first=True, dropout=self.DROPOUT, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(self.HIDDEN_SIZE * 2, self.OUTPUT_SIZE)
        else:
            self.fc = nn.Linear(self.HIDDEN_SIZE, self.OUTPUT_SIZE)

    def forward(self, x):
        if not x.dtype == torch.int:
            x = x.int()
        x = self.embedding(x)
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = x[:,-1,:]
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def init_hidden(self, size):
        if self.bidirectional:
            h0 = torch.zeros(self.N_LAYER * 2, size, self.HIDDEN_SIZE).requires_grad_()
        else:
            h0 = torch.zeros(self.N_LAYER, size, self.HIDDEN_SIZE).requires_grad_()
        return h0.detach(), h0.detach()


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)


class BERTTextClassifier(nn.Module):

    def __init__(self, NUM_CLASSES):
        super().__init__()
        self.BERT_MODEL_NAME = 'bert-base-cased'
        self.bert = BertModel.from_pretrained(self.BERT_MODEL_NAME)
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(self.bert.config.hidden_size, self.NUM_CLASSES)

    def forward(self, x):
        x = self.bert(x)
        output = self.fc(x.pooler_output)
        return output

