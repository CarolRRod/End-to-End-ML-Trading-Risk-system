import yfinance as yf
import pandas as pd

from typing import List

class MarketDataLoader:
    def __init__(self):
        pass

    def fetch_dataset(self, 
                      tickers: List[str],
                      start_date: str = "2018-01-01"):
        data = {}

        for ticker in tickers:
            df = yf.download(ticker,
                             start=start_date,
                             multi_level_index=False,
                             progress=False)
            
            if df.empty:
                continue
        
            df = self.validate_data(df)
            data[ticker] = df

        return df
    
    def validate_data(self, df: pd.DataFrame):
        df1 = df.copy()

        df1 = df1.sort_index()

        #Drop duplicated index
        df1 = df1[~df1.index.duplicated(keep="last")]

        # Reject if not enough data available
        if df1["Close"].isna().mean() > 0.05:
            return pd.DataFrame()
        
        # Drop rows with missing close values
        df1 = df1.dropna(subset=["Close"])

        # Min history of 3 years for stable predictions
        if len(df1) < (3*252):
            return pd.DataFrame()
        
        return df1
