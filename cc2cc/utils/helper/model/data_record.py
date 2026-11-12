import numpy as np
import pandas as pd

from cc2cc.utils.helper.units import AU2KCALMOL

TORCH_LIST = [
    "loss_ene",
    "loss_ene_abs",
    "loss_ene_atomic",
    "loss_grad_record",
    "loss_tot",
    "loss_tot_ene",
    "loss_tot_abs",
    "loss_tot_grad",
    "loss_tot_atomic",
]


class DataRecordList:
    def __init__(self, len_batch):
        self.data_dict = {
            "name": ["" for _ in range(len_batch)],
        }
        for key in TORCH_LIST:
            self.data_dict[key] = np.zeros(len_batch)
        self.iter = 0

    def add_data_record(self, data_record):
        for key, values in self.data_dict.items():
            if key not in data_record:
                raise ValueError(f"Key {key} not found in data_record.")
            value = data_record[key]
            if key == "name":
                values[self.iter] = value
            else:
                values[self.iter] = AU2KCALMOL * value.item()
        self.iter += 1

    def save(self, path):
        pd.DataFrame(self.data_dict).to_csv(path, index=False)

    def trim(self):
        self.data_dict = {
            key: values[: self.iter] for key, values in self.data_dict.items()
        }

    def merge(self):
        self.data_dict = (
            pd.DataFrame(self.data_dict)
            .groupby("name")
            .mean()
            .reset_index()
            .to_dict(orient="list")
        )
        self.iter = len(self.data_dict["name"])
