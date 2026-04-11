import torch
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    def __init__(self, panel, seq_len=60, target_col="returns_1d"):
        self.seq_len = seq_len

        self.samples = []

        for ticker, df in panel.groupby("Ticker"):
            df = df.sort_values("Date")

            X = df.drop(columns=["Date", "Ticker", target_col]).values
            y = df[target_col].values

            for i in range(len(df)-seq_len):
                X_seq = X[i:i+seq_len]
                y_seq = y[i+seq_len]

                self.samples((X_seq, y_seq))

        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X, y = self.samples[idx]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
