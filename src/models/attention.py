import torch 
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super.__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden)
        scores = self.attn(x)                   # (B, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights*x, dim=1)      # (B, hidden)


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_dim=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.norm =nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)


    def forward(self, x):
       lstm_out, _ = self.lstm(x)         # (batch_size, seq_len, hidden_size)
       lstm_out = self.norm(lstm_out)
       context = self.pool(lstm_out)      # (batch_size, hidden_size)
       out = self.fc(context)             # (batch_size, num_tickers)
       return out


class MultiHeadAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_heads=4, output_dim=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.pool = AttentionPooling(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
       lstm_out, _ = self.lstm(x)   # (batch_size, seq_len, hidden_size)

       attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)   # attn_out (batch_size, seq_len, hidden_size)
       out = self.norm(lstm_out, attn_output)
       context = self.pool(out)    # (batch_size, hidden_size)
       out = self.fc(context)      # (batch_size, output_dim)
       return out


class HybridAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_heads=4, output_dim=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_firs=True)
        self.pool = AttentionPooling(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) # (batch, seq_len, hidden_size)
        context = self.pool(attn_out)          # (batch, hidden_size)
        out = self.fc(context)                 # (batch, output_dim)

        return out
       