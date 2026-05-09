import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 bidirectional=False,
                 output_dim=1
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
        self.norm = nn.LayerNorm(hidden_size * direction_multiplier)
        self.fc = nn.Linear(hidden_size * direction_multiplier, output_dim)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """
        out, _ = self.gru(x)   # (batch_size, seq_len, hidden_size * direction_multiplier)
        out = out[:, -1]       # (batch_size, hidden_size * direction_multiplier)
        out = self.norm(out)
        out = self.fc(out)     # (batch_size, output_dim)
        return out


class Seq2SeqGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 horizon=10,
                 output_dim=1):
        super().__init__()


        self.horizon=horizon
        self.output_dim=output_dim

        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True)
        
        self.decoder_gru = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_size,
            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
         x: (batch_size, seq_len, input_size)
        """

        batch_size = x.size(0)
        _, hidden = self.encoder_gru(x)     # hidden: (batch_size, num_layers, hidden_size)

        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=x.device)
        outputs = []

        for _ in range(self.horizon):
            out, hidden = self.decoder_gru(decoder_input, hidden)   
            pred = self.fc(out)    # (batch_size, 1, output_dim)

            outputs.append(pred)
            decoder_input=pred

        
        outputs = torch.cat(outputs, dim=1) # (batch_size, horizon, output_dim)

        return outputs

