import torch
import torch.nn as nn

class MultiAssetLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 proj_size=0,
                 num_layers=1,
                 num_tickers=1,
                 bidirectional=False,
                 batch_first=True
                ):
        super().__init__()

        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                proj_size=proj_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
        
        lstm_out_size = proj_size if proj_size > 0 else hidden_size
        direction_multiplier = 2 if bidirectional else 1

        self.fc = nn.Linear(lstm_out_size * direction_multiplier, num_tickers)


    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        out, _ = self.lstm(x)   # (batch_size, seq_len, lstm_out_size*direction_multiplier)
        out = out[:, -1, :]     # (batch_size, lstm_out_size*direction_multiplier)
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
       

class MultiAssetSeq2SeqLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 horizon=10,
                 num_tickers=2):
        super().__init__()

        self.horizon = horizon
        self.num_tickers = num_tickers

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=num_tickers,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_tickers)

    
    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder_lstm(x)  # (batch_size, num_layers, hidden_size)
        decoder_input = torch.zeros(batch_size, 1, self.num_tickers).to(x.device)
        
        outputs = []

        for _ in range(self.horizon):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            decoder_input = self.fc(out)     # (batch_size, 1, num_tickers)
            
            outputs.append(decoder_input)

        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, num_tickers)

        return outputs


class MultiAssetSeq2SeqBiLSTM(MultiAssetSeq2SeqLSTM):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 horizon=10,
                 num_tickers=2):
        super().__init__(
            input_size,
            hidden_size=64,
            num_layers=2,
            horizon=10,
            num_tickers=2)

        self.horizon = horizon
        self.num_tickers = num_tickers
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    
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
        _, (hidden, cell) = self.encoder_lstm(x)  # (batch_size, 2*num_layers, hidden_size)

        # Merge bidirectional states
        hidden = self._merge_states(hidden)       # (batch_size, num_layers, hidden_size)
        cell = self._merge_states(cell)           # (batch_size, num_layers, hidden_size)
        
        decoder_input = torch.zeros(batch_size, 1, self.num_tickers).to(x.device)
        outputs = []

        for _ in range(self.horizon):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            decoder_input = self.fc(out)     # (batch_size, 1, num_tickers)
            
            outputs.append(decoder_input)

        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, num_tickers)

        return outputs

    