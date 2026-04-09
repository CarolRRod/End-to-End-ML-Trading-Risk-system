import torch 
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=1,
                 num_tickers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_tickers)


    def forward(self, x):
       """
         x: (batch_size, seq_len, input_size)
        """

       lstm_out, _ = self.lstm(x)         # (batch_size, seq_len, hidden_size)

       scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
       weights = F.softmax(scores, dim=1) # (batch_size, seq_len, 1)

       context = torch.sum(weights * lstm_out, dim=1)   # (batch_size, hidden_size)

       out = self.fc(context)   # (batch_size, num_tickers)
       return out


class MultiHeadAttentionLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_heads=4,
                 num_tickers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
           embed_dim=hidden_size,
           num_heads=num_heads,
           batch_first=True 
        )

        self.fc = nn.Linear(hidden_size, num_tickers)

    def forward(self, x):
       """x: (batch_size, seq_len, input_size)"""

       lstm_out, _ = self.lstm(x)   # (batch_size, seq_len, hidden_size)

       attn_output, _ = self.attention(
           lstm_out, lstm_out, lstm_out
       )        # attn_out (batch_size, seq_len, hidden_size)

       context = attn_output.mean(dim=1)    # (batch_size, hidden_size)
       out = self.fc(context)               # (batch_size, num_tickers)

       return out


class HybridAttentionLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_layers=2,
                 num_heads=4,
                 num_tickers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_firs=True
        )

        self.pooling = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, num_tickers)

    def forward(self, x):
        """
            x: (batch_size, seq_len, input_size)
        """

        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(lstm_out) # (batch, seq_len, hidden_size)
        scores = self.pooling(attn_out)        # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)     # (batch, seq_len, 1)
        context = torch.sum(attn_out * weights, dim=1) # (batch, hidden_size)

        out = self.fc(context)  # (batch, num_tickers)

        return out
    
