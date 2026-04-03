import torch.nn as nn
from src.models.base_model import BaseModel


class MultiAssetLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, num_tickers=1):
        super(MultiAssetLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size= hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_tickers)

    def forward(self, x):
        """
         x: (batch_size, seq_len, num_features)
        """

        out, _ = self.lstm(x)   # (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]     # (batch_size, 1, hidden_size)
        out = self.fc(out)      # (batch_size, num_tickers)
        return out