import pandas as pd

from src.data.dataloader import MarketDataLoader
from src.data.panel_builder import PanelBuilder

class FeatureEngineer:
    def __init__(self, panel: pd.DataFrame):
        self.panel = panel.copy()

    def returns(self, windows):
        for w in windows:
            self.panel[f"returns_{w}d"] = self.panel.groupby("Ticker")["Close"].pct_change(w)

    def volatility(self, windows):
        # Measures how much the price of an asset fluctuates over time
        if "returns_1d" not in self.panel.columns:
            self.returns(windows=(1, ))

        for w in windows:
            self.panel[f"volatility_{w}d"] = (self.panel.groupby("Ticker")["returns_1d"]
                                                .rolling(w)
                                                .std()
                                                .reset_index(level=0, drop=True))

    def beta(self, windows):
        """ Estimate the rolling beta for each ticker relative to SPY over a given period of time.
            Beta = Cov(stock, SPY) / Var(SPY)
        """

        if "returns_1d" not in self.panel.columns:
            self.returns(windows=(1, ))

        # Get market data (SPY)
        if "SPY" not in self.panel["Ticker"].unique():
            start_date = self.panel["Date"].min()
            spy_data = MarketDataLoader().fetch_dataset(["SPY"], start_date=start_date)
            spy_panel = PanelBuilder().build_panel(spy_data)

            fe = FeatureEngineer(spy_data)
            fe.returns((1,))
            spy_panel = fe.panel

            self.panel = pd.concat([self.panel, spy_panel])

        market =self.panel[self.panel["Ticker"] == "SPY"][["Date", "returns_1d"]].rename(columns={"returns_1d": "market_return"})

        self.panel = self.panel.merge(market, on="Date", how="left")

        for w in windows:
            cov = (self.panel.groupby("Ticker").apply(lambda x: x["returns_1d"].rolling(w).cov(x["market_return"])).reset_index(level=0, drop=True))
            var = (self.panel["market_return"].rolling(w).var())
            self.panel[f"beta_{w}d"] = cov/var

    def ma_ratio(self, windows):
        for w in windows:
            self.panel[f"ma_ratio_{w}d"] = (self.panel.groupby("Ticker")["Close"]
                                               .transform(lambda x: x / x.rolling(w).mean())
                                               )

    def transform(self, config):
        for func_name, windows in config.items():
            func = getattr(self, func_name, None)

            if func is None:
                raise ValueError (f"Warning feature {func_name} does not exist")
            
            func(windows)

        feature_cols = [col for col in self.panel.columns if any(k in col for k in config.keys())]
        feature_cols += ["Ticker", "Date"]
        panel = self.panel[feature_cols].dropna().sort_index(["Ticker", "Date"])

        return panel
