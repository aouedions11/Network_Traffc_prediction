import torch
import torch.nn as nn


class FBF_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=3):
        super(FBF_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=0.5)
        print(self.lstm)
        self.fc = nn.Linear(hidden_dim, 1)
        self.time_linear = nn.Linear(seq_len, pre_len)

    def forward(self, x):
        # BS,seq_len = x.size()
        # x = x.unsqueeze(-1)  # bs,t,1
        x, _ = self.lstm(x)  # BS,T,h
        x = self.fc(x)  # BS,T,1
        x = self.time_linear(x.squeeze(-1))
        return x
