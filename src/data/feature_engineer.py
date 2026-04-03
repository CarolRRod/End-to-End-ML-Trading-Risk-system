from src.data.dataloader import MarketDataLoader
from src.data.panel_builder import PanelBuilder

class FeatureEngineer:
    def __init__(self, panel):
        self.panel = panel

    def returns(self, windows):
        for days in windows:
            self.panel[f"returns_{days}d"] = self.panel.groupby(["Ticker"])["Close"].pct_change(days)

    def volatility(self, windows):
        # Measures how much the price of an asset fluctuates over time
        if "returns_1d" not in self.panel.columns:
            self.returns(windows=(1, ))

        for days in windows:
            self.panel[f"volatility_{days}d"] = (self.panel.groupby(["Ticker"])["returns_1d"]
                                                .rolling(days)
                                                .std()
                                                .reset_index(level=0, drop=True))

    def beta(self, windows):
        """ Estimate the rolling beta for each ticker relative to SPY over a given period of time.
            Beta = Cov(stock, SPY) / Var(SPY)
        """

        if "returns_1d" not in self.panel.columns:
            self.returns(windows=(1, ))

        # Get market data (SPY)
        if "SPY" in self.panel["Ticker"].unique():
            panel_spy = self.panel[self.panel["Ticker"] == "SPY"]
        else:
            start_date = self.panel["Date"].min()
            spy_data = MarketDataLoader().fetch_dataset(["SPY"], start_date=start_date)
            panel_spy = PanelBuilder().build_panel(spy_data)
            features_spy = FeatureEngineer(panel_spy)
            features_spy.returns((1, ))
            panel_spy = features_spy.panel

        market = panel_spy[panel_spy["Ticker"]=="SPY"][["Date", "returns_1d"]]
        market = market.rename(columns={"returns_1d" : "market_return"})

        self.panel = self.panel.merge(market, on="Date", how="left")
        self.panel = self.panel.dropna()

        for days in windows:
            self.panel[f"beta_{days}d"] = (self.panel.groupby(["Ticker"])
                                           .apply(
                                               lambda x : x["returns_1d"].rolling(days)
                                               .cov(x["market_return"])/
                                               x["market_return"].rolling(days).var()
                                           ).reset_index(level=0, drop=True)
                                           )

    def ma_ratio(self, windows):
        for days in windows:
            self.panel[f"ma_ratio_{days}d"] = (self.panel.groupby("Ticker")["Close"]
                                               .transform(lambda x: x / x.rolling(days).mean())
                                               )

    def transform(self, config):
        for func_name, windows in config.items():
            func = getattr(self, func_name, None)

            if func is not None:
                func(windows)
            else:
                print(f"Warning feature {func_name} does not exist")

        self.panel = self.panel.dropna()
        self.panel = self.panel.set_index("Date")

        feature_cols = [col for col in self.panel.columns if any(k in col for k in config.keys())]
        return self.panel[feature_cols].sort_index()
