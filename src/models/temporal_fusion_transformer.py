import torch 
import torch.nn as nn
import torch.nn.functional as F


class VariableSelection(nn.Module):
    # Learns which features matter at each timestep
    def __init__(self, input_size):
        super().__init__()

        self.varSelection = nn.Linear(input_size, input_size)

    def forward(self, x):
        # x : (batch, seq_len, input_size)
        scores = self.varSelection(x)        # (batch, seq_len, input_size)
        weights = F.softmax(scores, dim=1)   # (batch, seq_len, input_size)

        return x * weights 
    

class GRN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        residual = x        # (batch, seq_len, hidden_size)

        x = self.fc1(x)     # (batch, seq_len, hidden_size)
        x = F.elu(x)
        x = self.fc2(x)     # (batch, seq_len, hidden_size)

        gate = self.gate(x) # (batch, seq_len, hidden_size)
        gate = torch.sigmoid(gate)

        return gate * x + (1-gate) * residual

    
class MiniTFT(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_heads=4,
                 output_dim=2):
        super().__init__()

        self.var_select = VariableSelection(input_size)

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
        self.norm = nn.LayerNorm(hidden_size)
        self.grn = GRN(hidden_size, hidden_size)

        self.pool = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_dim)

    
    def forward(self, x):
        # x: (batch, seq_len, input_size)

        x = self.var_select(x)  # (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm(attn_out)

        gated = self.grn(attn_out)  # (batch, seq_len, hidden_size)
        gatedPool = self.pool(gated)    # (batch,seq_len,1)
        weights = torch.softmax(gatedPool, dim=1)
        context = torch.sum(gated*weights, dim=1) # (batch, hidden_size)
        out = self.fc(context)

        return out