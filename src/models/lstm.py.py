import torch.nn as nn

class MultiAssetLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 proj_size=0,
                 num_layers=1,
                 num_tickers=1,
                 bidirectional=False
                ):
        super().__init__()

        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                proj_size=proj_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        lstm_out_size = proj_size if proj_size > 0 else hidden_size
        direction_multiplier = 2 if bidirectional else 1

        self.fc = nn.Linear(lstm_out_size * direction_multiplier, num_tickers)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        out, _ = self.lstm(x)   # (batch_size, seq_len, lstm_out_size*direction_multiplier)
        out = out[:, -1, :]     # (batch_size, 1, lstm_out_size*direction_multiplier)
        out = self.fc(out)      # (batch_size, num_tickers)
        return out


class MultiAssetBiLSTM(MultiAssetLSTM):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_tickers=1):
        super().__init__(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_tickers=num_tickers,
            bidirectional=True
        )


class MultiAssetProjectedLSTM(MultiAssetLSTM):
    def __init__(self, input_size, hidden_size=64, num_layers=2, proj_size=32, num_tickers=1):
       super().__init__(
            input_size=input_size, 
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=num_layers,
            num_tickers=num_tickers,
        )