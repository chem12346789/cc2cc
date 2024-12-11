import pandas as pd


class Data_Record:

    def __init__(self, path):
        self.df_dict = {"name": []}
        self.path = path

    def add_data(self, name, dict_: dict):
        """
        Add data to the dictionary
        """
        if isinstance(name, list):
            for n in name:
                self.df_dict["name"].append(n)
        else:
            self.df_dict["name"].append(name)

        for key, val in dict_.items():
            if key not in self.df_dict:
                self.df_dict[key] = []
            if isinstance(val, list):
                for v in val:
                    self.df_dict[key].append(v)
            else:
                self.df_dict[key].append(val)

    def save_csv(self):
        """
        save the loss to a csv file
        """
        df = pd.DataFrame(self.df_dict)
        df.to_csv(self.path, index=False)
