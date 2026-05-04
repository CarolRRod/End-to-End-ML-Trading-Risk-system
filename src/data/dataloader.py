import yfinance as yf
import pandas as pd
from typing import List

class MarketDataLoader:
    def __init__(self, min_years: int = 3):
        self.min_days = 252 * min_years

    def fetch_dataset(self, tickers: List[str], start_date: str = "2018-01-01"):
        data = {}

        for ticker in tickers:
            df = yf.download(ticker,
                             start=start_date,
                             multi_level_index=False,
                             progress=False,
                             auto_adjust=True)

            if df.empty:
                continue

            df = self.validate_data(df)
            
            if df.empty:
                continue

            data[ticker] = df

        return data

    def validate_data(self, df: pd.DataFrame):
        df = df.sort_index()
        #Drop duplicated index
        df = df[~df.index.duplicated(keep="last")]

        if "Close" not in df.columns:
            return pd.DataFrame()

        # Reject if not enough data available
        if df["Close"].isna().mean() > 0.05:
            return pd.DataFrame()

        # Drop rows with missing close values
        df = df.dropna(subset=["Close"])

        # Min history of min_days for stable predictions
        if len(df) < self.min_days:
            return pd.DataFrame()

        return df
