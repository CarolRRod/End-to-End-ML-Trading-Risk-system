import torch
import numpy as np

from torch.utils.data import Dataset

class MarketDataset(Dataset):
    def __init__(self, panel, seq_len=60, target_col="returns_1d"):
        self.seq_len = seq_len
        self.target_col = target_col

        if target_col not in panel.columns:
            raise ValueError(f"{target_col} not found in panel")
        
        self.samples = []
        self.data = {}

        for ticker, df in panel.groupby("Ticker"):
            df = df.sort_values("Date")

            X = df.drop(columns=["Date", "Ticker", target_col]).values.astype(np.float32)
            y = df[target_col].values.astype(np.float32)

            self.data[ticker] = (X, y)

            for i in range(len(df)-seq_len):
                self.samples.append((ticker, i))

        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ticker, i = self.samples[idx]
        X, y = self.data[ticker]

        X_seq = X[i : i + self.seq_len]
        y_seq = y[i + self.seq_len]

        return (
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq)
        )