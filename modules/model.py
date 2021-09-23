import torch
from torch import nn
from torch.nn import functional as F


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
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = x[:,-1,:]
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def init_hidden(self, size):
        hidden = torch.zeros(self.N_LAYER, size, self.HIDDEN_SIZE).requires_grad_()
        return hidden.detach(), hidden.detach()




