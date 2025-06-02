import pandas as pd
import numpy as np


class DataRecord:

    def __init__(self, path, if_continue=False):
        self.df_dict = {"name": []}
        self.path = path
        if if_continue:
            try:
                df = pd.read_csv(path)
                for key in df.keys():
                    self.df_dict[key] = df[key].tolist()
            except FileNotFoundError:
                pass

    def add_data(self, dict_: dict):
        """
        Add data to the dictionary
        """
        for key, val in dict_.items():
            if key not in self.df_dict:
                self.df_dict[key] = []
            else:
                self.df_dict[key].append(val)

    def save_csv(self):
        """
        save the loss to a csv file
        """
        df = pd.DataFrame(self.df_dict)
        df.to_csv(self.path, index=False)
