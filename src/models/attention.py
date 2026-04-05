import torch 
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=1,
                 num_tickers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_tickers)


    def forward(self, x):
       """
         x: (batch_size, seq_len, input_size)
        """

       lstm_out, _ = self.lstm(x)         # (batch_size, seq_len, hidden_size)

       scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
       weights = F.softmax(scores, dim=1) # (batch_size, seq_len, 1)

       context = torch.sum(weights * lstm_out, dim=1)   # (batch_size, hidden_size)

       out = self.fc(context)   # (batch_size, num_tickers)
       return out
