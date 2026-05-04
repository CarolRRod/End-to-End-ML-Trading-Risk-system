import pandas as pd
from typing import Dict

class PanelBuilder:
    def build_panel(self, data_dict: Dict[str, pd.DataFrame]):
        frames = []

        for ticker, df in data_dict.items():
            temp = df.copy()
            temp["Ticker"] = ticker
            temp["Date"] = temp.index
            frames.append(temp)

        panel = pd.concat(frames, axis=0, ignore_index=True)
        panel = panel.sort_values(["Ticker", "Date"])

        return panel
