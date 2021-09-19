import torch
from torch import nn


class TextClassifierLSTM(nn.Module):

    def __init__(self, VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE, NUM_CLASSES, N_LAYER):
        super(TextClassifierLSTM, self).__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMB_SIZE = EMB_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        self.N_LAYER = N_LAYER
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.EMB_SIZE)
        self.lstm = nn.LSTM(self.EMB_SIZE, self.HIDDEN_SIZE, num_layers=self.N_LAYER, batch_first=True)
        self.fc = nn.Linear(self.HIDDEN_SIZE, self.NUM_CLASSES)

    def forward(self, x):
        if not x.dtype == torch.int:
            x = x.int()
        x = self.embedding(x)
        x, _ = self.lstm(x, self.init_hidden())
        x = x.squeeze()
        x = self.fc(x)
        return x

    def init_hidden(self):
        hidden = torch.zeros(self.N_LAYER, 1, self.HIDDEN_SIZE).requires_grad_()
        return hidden.detach(), hidden.detach()


