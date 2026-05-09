import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 output_size=1,
                 bidirectional=False,
                ):
        super().__init__()

        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        direction_multiplier = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * direction_multiplier)
        self.fc = nn.Linear(hidden_size * direction_multiplier, output_size)


    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        out, _ = self.lstm(x)   # (batch_size, seq_len, lstm_out_size*direction_multiplier)
        out = out[:, -1]        # (batch_size, lstm_out_size*direction_multiplier)
        out = self.norm(out)
        out = self.fc(out)      # (batch_size, output_size)
        return out


class Seq2SeqLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 horizon=10,
                 output_size=1):
        super().__init__()

        self.horizon = horizon
        self.output_size = output_size

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder_lstm(x)  # (num_layers, batch_size, hidden_size)
        decoder_input = torch.zeros(batch_size, 1, self.num_tickers).to(x.device)
        
        outputs = []

        for _ in range(self.horizon):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            pred = self.fc(out)     # (batch_size, 1, output_size)
            
            outputs.append(decoder_input)
            decoder_input = pred

        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, output_size)

        return outputs


class Seq2SeqBiLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 horizon=10,
                 output_size=1):
        super().__init__()

        self.horizon = horizon
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_size*2, output_size)

    
    def _merge_states(self, x):

        x = x.view(self.num_layers, 2, x.size(1), self.hidden_size)
        x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=2)    # (num_layers, batch_size, hidden_size*2)
        return x

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder_lstm(x)  # (2*num_layers, batch_size, hidden_size)

        # Merge bidirectional states
        hidden = self._merge_states(hidden)      # (num_layers, batch_size, hidden_size*2)
        cell = self._merge_states(cell)          # (num_layers, batch_size, hidden_size*2)
        
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device= x.device)
        outputs = []

        for _ in range(self.horizon):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            pred = self.fc(out)     
            outputs.append(decoder_input)
            decoder_input = pred

        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, output_dim)

        return outputs
