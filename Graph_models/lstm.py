import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        # BS,seq_len,f = x.size()
        x, _ = self.lstm(x)  # BS,T,h
        x = self.fc(x)  # BS,T,f
        return x[:, -1]
