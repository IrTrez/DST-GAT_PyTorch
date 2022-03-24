import torch
from torch import nn
from torch import functional as F


class lstmTorch(nn.Module):
    def __init__(self, Nfeatures=12, lookAhead=6, dropoutAmount=0.25, LSTMSize=200, batchSize=64):
        super(lstmTorch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=Nfeatures, hidden_size=200, num_layers=3, batch_first=True
        )
        self.dense1 = nn.Linear(LSTMSize, LSTMSize)

        self.dropoutAmount = dropoutAmount
        self.dropoutt = nn.Dropout(self.dropoutAmount)

        self.relu = nn.ReLU()

        self.dense2 = nn.Linear(LSTMSize, int(LSTMSize/2))
        self.dense3 = nn.Linear(int(LSTMSize/2), 12)

        self.lookAhead = lookAhead
        self.batchSize = batchSize

    def forward(self, x):
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.relu(x)
        x = self.dense1(x)
        x = self.dropoutt(x)
        x = self.dense2(x)
        x = self.dense3(x)


        x = torch.reshape(x, (self.batchSize, self.lookAhead, 2))

        return x