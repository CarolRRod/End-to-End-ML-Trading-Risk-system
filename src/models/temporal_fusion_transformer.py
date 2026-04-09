import torch 
import torch.nn as nn
import torch.nn.functional as F


class VariableSelection(nn.Module):
    # Learns which features matter at each timestep
    def __init__(self, input_size, num_features):
        super().__init__()

        self.varSelection = nn.Linear(input_size, num_features)

    def forward(self, x):
        # x : (batch, seq_len, input_size)

        scores = self.varSelection(x)        # (batch, seq_len, num_features)
        weights = F.softmax(scores, dim=1)    # (batch, seq_len, num_features)

        return x * weights
    

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.gate = nn.Linear(input_size, input_size)

    def forward(self, x):
        residual = x        # (batch, seq_len, input_size)

        x = self.fc1(x)     # (batch, seq_len, hidden_size)
        x = F.elu(x)
        x = self.fc2(x)     # (batch, seq_len, input_size)

        gate = self.gate(x)
        gate = torch.sigmoid(gate)

        return gate * x + (1-gate) * residual

    
class MiniTFT(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_heads=4,
                 num_tickers=2):
        super().__init__()

        self.var_select = VariableSelection(input_size, input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_fist=True
        )

        self.grn = GRN(hidden_size, hidden_size)

    
    def forward(self, x):
        # x: (batch, seq_len, input_size)

        x = self.var_select(x)  # (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        gated = self.grn(attn_out)  # (batch, seq_len, hidden_size)
        context = gated.mean(dim=1)
        out = self.fc(context)

        return context
