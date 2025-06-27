"""
A class to record data in a dictionary and save it to a CSV file.
"""

import pandas as pd


class DataRecord:
    """
    A class to record data in a dictionary and save it to a CSV file.
    The data is stored in a dictionary where keys are column names and values are lists of data
    """

    def __init__(self, path, if_continue=False):
        self.df_dict = {}
        self.path = path
        self.length = 0

        if if_continue:
            try:
                df = pd.read_csv(path)
                for key in df.keys():
                    self.df_dict[key] = df[key].tolist()
                    self.length = len(self.df_dict[key])
            except FileNotFoundError:
                pass

    def add_data(self, dict_: dict):
        """
        Add data to the dictionary
        """
        for key, val in dict_.items():
            if key not in self.df_dict:
                self.df_dict[key] = [None] * self.length
            self.df_dict[key].append(val)
        self.length += 1

    def save_csv(self):
        """
        save the loss to a csv file
        """
        df = pd.DataFrame(self.df_dict)
        df.to_csv(self.path, index=False)
