import torch.nn as nn

class MultiAssetGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 bidirectional=False,
                 num_tickers=1
                ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_multiplier, num_tickers)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """
        out, _ = self.gru(x)   # (batch_size, seq_len, hidden_size * direction_multiplier)
        out = out[:, -1, :]    # (batch_size, 1, hidden_size * direction_multiplier)
        out = self.fc(out)
        return out


class MultiAssetBiGRU(MultiAssetGRU):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 bidirectional=True,
                 num_tickers=1):
        super().__init__(input_size, hidden_size, num_layers, bidirectional, num_tickers)
