import torch
from torch import nn
from torch.nn import functional as F


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




