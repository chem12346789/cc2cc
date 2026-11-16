import numpy as np
import pandas as pd

from cc2cc.utils.helper.units import AU2KCALMOL

SCHEMA_VERSION = 2

TORCH_LIST = [
    "energy_abs_error_kcal_mol",
    "grid_weighted_abs_error_kcal_mol",
    "atomic_abs_error_kcal_mol",
    "force_abs_error_sum",
    "objective_total_raw",
    "objective_energy_raw",
    "objective_grid_raw",
    "objective_force_raw",
    "objective_atomic_raw",
    "zmp_response_abs_error",
    "objective_zmp_response_raw",
]

TIMING_COLUMNS = [
    "time_energy_s",
    "time_grid_s",
    "time_force_s",
    "time_atomic_s",
]
KCALMOL_ERRORS = {
    "energy_abs_error_kcal_mol",
    "grid_weighted_abs_error_kcal_mol",
    "atomic_abs_error_kcal_mol",
}


class DataRecordList:
    def __init__(self, len_batch):
        self.data_dict = {
            "schema_version": np.full(len_batch, SCHEMA_VERSION, dtype=np.int64),
            "sample_name": ["" for _ in range(len_batch)],
        }
        for key in TORCH_LIST + TIMING_COLUMNS:
            self.data_dict[key] = np.full(len_batch, np.nan)
        self.iter = 0

    def add_data_record(self, data_record):
        for key, values in self.data_dict.items():
            if key == "schema_version":
                continue
            if key in TIMING_COLUMNS:
                values[self.iter] = np.nan
                continue
            if key not in data_record:
                raise ValueError(f"Key {key} not found in data_record.")
            value = data_record[key]
            if key == "sample_name":
                values[self.iter] = value
            else:
                value = value.item()
                values[self.iter] = (
                    AU2KCALMOL * value if key in KCALMOL_ERRORS else value
                )
        self.iter += 1

    def save(self, path):
        pd.DataFrame(self.data_dict).to_csv(path, index=False)

    def trim(self):
        self.data_dict = {
            key: values[: self.iter] for key, values in self.data_dict.items()
        }

    def merge(self):
        columns = list(self.data_dict)
        self.data_dict = (
            pd.DataFrame(self.data_dict)
            .groupby("sample_name")
            .mean()
            .reset_index()
        )
        self.data_dict["schema_version"] = SCHEMA_VERSION
        self.data_dict = self.data_dict[columns].to_dict(orient="list")
        self.iter = len(self.data_dict["sample_name"])
