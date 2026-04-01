import pandas as pd 


class PanelBuilder:
    def __init__(self):
        pass

    def build_panel(self, data_dict):
        frames = []

        for ticker, df in data_dict.items():
            temp = df.copy()
            temp["Ticker"] = ticker
            temp["Date"] = temp.index

            frames.append(temp)

        panel = pd.concat(frames)
        panel = panel.reset_index(drop=True)
        panel = panel.sort_values(["Date", "Ticker"])

        return panel
        
