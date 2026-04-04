import torch
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
            bidirectional=bidirectional,
            batch_first=True
        )

        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_multiplier, num_tickers)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """
        out, _ = self.gru(x)   # (batch_size, seq_len, hidden_size * direction_multiplier)
        out = out[:, -1, :]    # (batch_size, hidden_size * direction_multiplier)
        out = self.fc(out)     # (batch_size, num_tickers)
        return out


class MultiAssetBiGRU(MultiAssetGRU):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 bidirectional=True,
                 num_tickers=1):
        super().__init__(input_size, hidden_size, num_layers, bidirectional, num_tickers)


class MultiAssetSeq2SeqGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_tickers=2,
                 horizon=10):
        super().__init__()

        self.horizon=horizon
        self.num_tickers=num_tickers

        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        
        self.decoder_gru = nn.GRU(
            input_size=num_tickers,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_tickers)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        batch_size = x.size(0)
        _, hidden = self.encoder_gru(x)     # hideden: (batch_size, num_layers, hidden_size)

        decoder_input = torch.zeros(batch_size, 1, self.num_tickers).to(x.device)
        outputs = []

        for _ in range(self.horizon):
            out, hidden = self.decoder_gru(decoder_input, hidden)   
            decoder_input = self.fc(out)    # (batch_size, 1, num_tickers)

            outputs.append(decoder_input)
        
        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, num_tickers)

        return outputs

class MultiAssetBiGRU(MultiAssetGRU):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 bidirectional=True,
                 num_tickers=1):
        super().__init__(input_size, hidden_size, num_layers, bidirectional, num_tickers)


class MultiAssetSeq2SeqGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_tickers=2,
                 horizon=10):
        super().__init__()

        self.horizon=horizon
        self.num_tickers=num_tickers

        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        
        self.decoder_gru = nn.GRU(
            input_size=num_tickers,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_tickers)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        batch_size = x.size(0)
        _, hidden = self.encoder_gru(x)     # hidden: (batch_size, num_layers, hidden_size)

        decoder_input = torch.zeros(batch_size, 1, self.num_tickers).to(x.device)
        outputs = []

        for _ in range(self.horizon):
            out, hidden = self.decoder_gru(decoder_input, hidden)   
            decoder_input = self.fc(out)    # (batch_size, 1, num_tickers)

            outputs.append(decoder_input)
        
        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, num_tickers)

        return outputs


class MultiAssetSeq2SeqBiGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_tickers=2,
                 horizon=10):
        super().__init__(
            input_size,
            hidden_size=64,
            num_layers=2,
            num_tickers=2,
            horizon=10
        )

        self.horizon=horizon
        self.num_tickers=num_tickers
        self.num_layers=num_layers
        self.hidden_size=hidden_size

        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)

    def _merge_states(self, x):
        """
         x: (batch_size, 2 * num_layers, hidden_size)
        """

        x = x.view(x.size(0), self.num_layers, 2, self.hidden_size)
        x = x.sum(dim=1)    # (batch_size, num_layers, hidden_size)

        return x

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        batch_size = x.size(0)
        _, hidden = self.encoder_gru(x)     # hidden: (batch_size, 2*num_layers, hidden_size)

        # Merge bidirectional states
        hidden = self._merge_states(hidden)       # (batch_size, num_layers, hidden_size)
        
        decoder_input = torch.zeros(batch_size, 1, self.num_tickers).to(x.device)
        outputs = []

        for _ in range(self.horizon):
            out, hidden = self.decoder_gru(decoder_input, hidden)   
            decoder_input = self.fc(out)    # (batch_size, 1, num_tickers)

            outputs.append(decoder_input)
        
        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, num_tickers)

        return outputs

