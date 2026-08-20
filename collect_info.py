from pathlib import Path
import json
import time
import argparse
import os

import pandas as pd
import numpy as np
import wandb

AU2KCALMOL = 627.5094740631
DIET_DATASETS = {"gmtkn-diet30-def2", "gmtkn-diet100-def2"}
DATA_FRAME_NAMES = {
    "gmtkn-def2": [
        "subset_mean_df",
        "subset_wtmad_2_df",
        "wtmad_1_df",
        "wtmad_2_df",
    ],
    "dft-fitset-def2": ["subset_rmsd_df", "subset_mapd_df", "subset_mpd_df"],
    "gmtkn-diet30-def2": ["subset_mean_df", "subset_wtmad_2_df", "wtmad_2_df"],
    "gmtkn-diet100-def2": ["subset_mean_df", "subset_wtmad_2_df", "wtmad_2_df"],
}
RENAME_DICT = {
    "cc_ene": "mae",
    "b3lyp_ene": "B3LYP",
    "b3lyp-d3bj_ene": "B3LYP(BJ)",
    "b3lyp-d3zero_ene": "B3LYP(0)",
    "scf_ene_sota": "SOTA",
    "scf_d3bj_ene_sota": "SOTA(BJ)",
    "scf_ene": "AI",
    "scf_d3bj_ene": "AI(BJ)",
    "modified_ai_d3bj": "AI(M BJ)",
}


def divide(a, b):
    result = np.zeros_like(b, dtype=float)
    np.divide(a, b, out=result, where=(b != 0))
    return result


class Collect_info:
    def __init__(
        self,
        model_load,
        epoch,
        basis,
        verbose=4,
        data_set="gmtkn-def2",
    ):
        self.model_load = model_load
        self.epoch = epoch
        self.basis = basis
        self.verbose = verbose
        self.data_set = data_set
        self.data_frame_name_list = DATA_FRAME_NAMES[self.data_set]
        self.data_frame_dict = {}

        dir_str = os.environ.get("VALIDATE_DIR", "validate_hkqai")
        self.stamp = f"{self.basis}_{self.model_load}_{self.epoch}_{self.data_set}"
        self.data_path = Path(dir_str) / f"ccdft_{self.stamp}.csv"
        self.done_cache_path = Path(dir_str) / "collect_done_cache.json"
        self.if_done = self.load_done_cache()

        if self.if_done:
            self.run = None
            cache_entry = self._get_done_cache_entry()
            output_log_str = cache_entry.get("output_log_str", "")
            if output_log_str:
                print(output_log_str, end="" if output_log_str.endswith("\n") else "\n")
            else:
                print("DONE")
        else:
            experiment_dict = {
                "model_load": self.model_load,
                "epoch": self.epoch,
                "basis": self.basis,
                "verbose": self.verbose,
                "pid": os.getpid(),
            }
            self.run = wandb.init(
                project="DFT2CC_validation",
                resume="allow",
                name=(f"collect_{self.stamp}"),
                config=experiment_dict,
                allow_val_change=True,
                mode="disabled" if self.verbose < 3 else "online",
            )

            print("Collect_info initialized with the following parameters:")
            for key, value in experiment_dict.items():
                print(f"{key}: {value}")
            print("")

        with open(
            "/home/chenzihao/workspace/cc2cc_test5/cc2cc/utils/mol_dataset/subset.json"
        ) as f:
            self.full_subset_dict = json.load(f)[self.data_set]

        for name_set, subset_list_ in self.full_subset_dict.items():
            self.full_subset_dict[name_set] = np.sort(subset_list_).tolist()

        self.name_subset_list = [
            name_subset
            for subset in self.full_subset_dict.values()
            for name_subset in subset
        ]
        self.name_set_list = list(self.full_subset_dict.keys())
        self.data = pd.DataFrame()

    def _load_done_cache_map(self):
        if not self.done_cache_path.exists():
            return {}
        try:
            with open(self.done_cache_path, "r") as f:
                cache = json.load(f)
            return cache if isinstance(cache, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _get_done_cache_entry(self):
        cache = self._load_done_cache_map()
        entry = cache.get(self.stamp, False)
        if isinstance(entry, dict):
            return entry
        return {"done": bool(entry), "output_log_str": ""}

    def load_done_cache(self):
        return bool(self._get_done_cache_entry().get("done", False))

    def save_done_cache(self, done, output_log_str=""):
        if not done:
            return
        cache = self._load_done_cache_map()
        entry = cache.get(self.stamp, False)
        if isinstance(entry, dict) and entry.get("done", False):
            return
        if isinstance(entry, bool) and entry:
            return
        cache[self.stamp] = {"done": True, "output_log_str": output_log_str}
        self.done_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.done_cache_path, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)

    def reset(self):
        self.data = pd.DataFrame()
        if self.verbose > 3:
            print("Data reset.")

    def aggregate_data(self):
        for name_subset in self.name_subset_list:
            if "gmtkn-diet" in args.data_set.lower():
                pattern = f"*{self.stamp}_gmtkn-def2_molecule_{name_subset}.csv"
            else:
                pattern = f"*{self.stamp}_molecule_{name_subset}.csv"
            print(f"Find {pattern} in validate directory...")
            data_path_list = list(Path("validate").glob(pattern))
            if len(data_path_list) != 1:
                if self.verbose > 3:
                    print(
                        "Warning: Expected 1 file for "
                        f"{name_subset} but found {len(data_path_list)}"
                    )
                continue
            source_data_path = data_path_list[0]

            with open(source_data_path, "r") as f:
                self.data = pd.concat([self.data, pd.read_csv(f)], ignore_index=True)

        if self.data.empty:
            print("No data found to aggregate.")

    @staticmethod
    def _subset_alias(subset_name):
        if subset_name == "BH76RC":
            return "BH76"
        for prefix in ("S66x8", "S22x5"):
            if prefix in subset_name:
                return prefix
        return subset_name

    @staticmethod
    def _format_status(done, total):
        return np.array(
            [
                (
                    f"{int(done_i)}/{int(total_i)}"
                    if int(done_i) < int(total_i)
                    else "DONE"
                )
                for done_i, total_i in zip(done, total)
            ],
            dtype=object,
        )

    def _update_wtmad2_frames(self, dft_type, wtmad_2_subset, subset2set):
        self.data_frame_dict["subset_wtmad_2_df"].loc[
            self.name_subset_list, dft_type
        ] = wtmad_2_subset
        set_wtmad_2 = np.einsum("i,ij->j", wtmad_2_subset, subset2set)
        self.data_frame_dict["wtmad_2_df"].loc[
            self.name_set_list, dft_type
        ] = set_wtmad_2
        self.data_frame_dict["wtmad_2_df"].loc["summary", dft_type] = np.sum(
            set_wtmad_2
        )

    def add_d3bj_correction(self):
        print(f"Current data columns: {self.data.columns.tolist()}")
        if (
            self.data.empty
            or "scf_d3bj_ene" in self.data.columns
            or "b3lyp-d3bj_ene" in self.data.columns
        ):
            if self.verbose > 3:
                print("D3BJ correction already added.")
            return

        if "name" not in self.data.columns:
            return

        reference_set = (
            "gmtkn-def2" if self.data_set in DIET_DATASETS else self.data_set
        )
        data_test = pd.read_csv(
            f"validate_hkqai_done/ccdft_def2-QZVP(D)__{reference_set}.csv"
        )
        correction_by_name = dict(
            zip(data_test["name"], data_test["b3lyp-d3bj_ene"] - data_test["b3lyp_ene"])
        )

        reference_names = self.data["name"].str.replace(
            self.basis, "def2-QZVP(D)", regex=False
        )
        correction = reference_names.map(correction_by_name)
        if correction.isna().any() and self.verbose > 3:
            missing = np.unique(reference_names[correction.isna()].to_numpy())
            print(f"Warning: Missing D3BJ references for {missing.tolist()}")
        self.data["scf_d3bj_ene"] = self.data["scf_ene"] + correction.to_numpy()

    def get_wtmad_2(self):
        output_log_list = []

        def log_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            output_log_list.append(sep.join(str(arg) for arg in args) + end)
            return print(*args, **kwargs)

        if self.data.empty:
            if self.verbose > 3:
                log_print("No data available to calculate WTMAD-2.")
            self.if_done = False
            return False

        data_name_list = (
            self.data["name"].str.split(f"_{self.basis}", regex=False).str[0].to_numpy()
        )
        with open(
            f"/home/chenzihao/workspace/cc2cc_test5/cc2cc/utils/mol_dataset/{self.data_set}.json"
        ) as f:
            data_set_json = json.load(f)

        reference_energy = []
        molecules_to_reactions = []
        reactions_to_subset = []
        total_counts_subset = []
        weight_list = []

        for i_subset, subset_name in enumerate(self.name_subset_list):
            i_subset_name = self._subset_alias(subset_name)
            log_print(subset_name, i_subset_name)

            reaction_dict = data_set_json[f"reaction-{subset_name}"]
            total_counts_subset.append(len(reaction_dict))
            # Use the first reaction since all reactions in a subset share weight.
            first_reaction = next(iter(reaction_dict.values()))
            if "weight" in first_reaction:
                weight_list.append(first_reaction["weight"])

            for i_reaction in reaction_dict.values():
                systems_list = i_reaction["systems"]
                stoichiometry_list = i_reaction["stoichiometry"]
                molecule_stoichiometry = np.zeros(len(data_name_list))
                finished = True

                for i in range(len(systems_list)):
                    mole_name = f"{i_subset_name}-{systems_list[i]}"
                    stoichiometry = int(stoichiometry_list[i])

                    if mole_name in data_set_json:
                        if isinstance(data_set_json[mole_name], str):
                            mole_name = data_set_json[mole_name]

                    col = np.where(data_name_list == mole_name)[0]
                    if col.size == 1:
                        molecule_stoichiometry[col[0]] = stoichiometry
                    else:
                        if self.verbose > 3:
                            log_print(
                                "Warning: Could not find molecule "
                                f"{mole_name} in subset {subset_name}"
                            )
                        finished = False

                if finished:
                    reference_energy.append(i_reaction["reference"])
                    molecules_to_reactions.append(molecule_stoichiometry)
                    subset_index = np.zeros(len(self.name_subset_list))
                    subset_index[i_subset] = 1
                    reactions_to_subset.append(subset_index)

        subset2set = np.array(
            [
                [
                    subset_name in subset_list
                    for subset_list in self.full_subset_dict.values()
                ]
                for subset_name in self.name_subset_list
            ],
            dtype=float,
        )

        reference_energy = np.array(reference_energy)
        molecules_to_reactions = np.array(molecules_to_reactions)
        reactions_to_subset = np.array(reactions_to_subset)
        completed_counts_subset = np.sum(reactions_to_subset, axis=0)
        total_counts_subset = np.array(total_counts_subset)
        weight_list = np.array(weight_list)

        dft_type_list = (
            ["b3lyp_ene", "b3lyp-d3bj_ene", "b3lyp-d3zero_ene"]
            if self.model_load == ""
            else ["scf_ene", "scf_d3bj_ene", "modified_ai_d3bj"]
        )
        header = dft_type_list + ["Processed"]

        for data_frame_name in self.data_frame_name_list:
            if Path(
                f"validate_hkqai_done/{data_frame_name}_{self.data_set}.csv"
            ).exists():
                df = pd.read_csv(
                    f"validate_hkqai_done/{data_frame_name}_{self.data_set}.csv",
                    index_col=0,
                )
                for col in header:
                    if col not in df.columns:
                        df[col] = 0.0
                # move Processed column to the end
                df.pop("Processed")
            else:
                print(
                    "DataFrame "
                    f"{data_frame_name}_{self.data_set} not found. "
                    "Initializing new DataFrame."
                )
                if data_frame_name.startswith("subset"):
                    df = pd.DataFrame(
                        index=self.name_subset_list, columns=dft_type_list
                    )
                else:
                    df = pd.DataFrame(
                        index=self.name_set_list + ["summary"], columns=dft_type_list
                    )
            self.data_frame_dict[data_frame_name] = df

        status_subset = self._format_status(
            completed_counts_subset, total_counts_subset
        )
        log_print(f"Number of reactions process/done in each subset: {status_subset}")

        completed_counts_set = (status_subset == "DONE").astype(int)
        completed_counts_set = np.einsum("i,ij->j", completed_counts_set, subset2set)
        total_counts_set = np.einsum("i,ij->j", np.ones_like(status_subset), subset2set)
        status_set = self._format_status(completed_counts_set, total_counts_set)
        log_print(
            f"Number of reactions process/done in each set: {completed_counts_set}"
        )

        status_summary = self._format_status(
            [np.sum(completed_counts_set)], [np.sum(total_counts_set)]
        )[0]
        log_print(f"Total number of reactions process/done: {status_summary}")

        for df_name in self.data_frame_name_list:
            df = self.data_frame_dict[df_name]
            if df_name.startswith("subset"):
                df.loc[self.name_subset_list, "Processed"] = status_subset
            else:
                df.loc[self.name_set_list, "Processed"] = status_set
                df.loc["summary", "Processed"] = status_summary

        # mae: mean_relative_abs_energies
        mean_relative_abs_energies = np.einsum(
            "i,ij,j->j",
            np.abs(reference_energy),
            reactions_to_subset,
            divide(1, completed_counts_subset),
        )
        inverse_mae = divide(1, mean_relative_abs_energies)

        if self.model_load == "" and self.data_set == "gmtkn-def2":
            self.data_frame_dict["subset_mean_df"].loc[
                self.name_subset_list, "mae"
            ] = mean_relative_abs_energies
        mean_absolute_deviation = 56.84 / 1505

        for dft_type in dft_type_list:
            if dft_type not in self.data.columns:
                if self.verbose > 3:
                    log_print(f"Warning: {dft_type} not found in data columns.")
                for df in self.data_frame_dict.values():
                    df.pop(dft_type)
                continue
            data_dft_ene = self.data[dft_type].to_numpy() * AU2KCALMOL

            reaction_energy_dft = np.einsum(
                "ji,i->j",
                molecules_to_reactions,
                data_dft_ene,
            )
            mean_reaction_energy = np.einsum(
                "i,ij,j->j",
                np.abs(reference_energy - reaction_energy_dft),
                reactions_to_subset,
                divide(1, completed_counts_subset),
            )

            if self.data_set == "gmtkn-def2":
                wtmad_1_multiplier = np.ones_like(mean_relative_abs_energies)
                wtmad_1_multiplier[mean_relative_abs_energies > 75] = 0.1
                wtmad_1_multiplier[mean_relative_abs_energies < 7.5] = 10

                wtmad_1_subset = wtmad_1_multiplier * mean_reaction_energy
                self.data_frame_dict["wtmad_1_df"].loc[self.name_set_list, dft_type] = (
                    np.einsum("i,ij->j", wtmad_1_subset, subset2set)
                    / len(self.name_subset_list)
                )
                self.data_frame_dict["wtmad_1_df"].loc["summary", dft_type] = np.sum(
                    self.data_frame_dict["wtmad_1_df"].loc[self.name_set_list, dft_type]
                )
                self.data_frame_dict["subset_mean_df"].loc[
                    self.name_subset_list, dft_type
                ] = mean_reaction_energy
                wtmad_2_subset = mean_absolute_deviation * np.einsum(
                    "i,i,i->i",
                    total_counts_subset,
                    inverse_mae,
                    mean_reaction_energy,
                )
                self._update_wtmad2_frames(dft_type, wtmad_2_subset, subset2set)
            elif self.data_set == "dft-fitset-def2":
                delta = reference_energy - reaction_energy_dft
                self.data_frame_dict["subset_mpd_df"].loc[
                    self.name_subset_list, dft_type
                ] = 100 * np.einsum(
                    "i,ij,j,j->j",
                    delta,
                    reactions_to_subset,
                    inverse_mae,
                    divide(1, completed_counts_subset),
                )
                self.data_frame_dict["subset_mapd_df"].loc[
                    self.name_subset_list, dft_type
                ] = 100 * np.einsum(
                    "i,ij,j,j->j",
                    np.abs(delta),
                    reactions_to_subset,
                    inverse_mae,
                    divide(1, completed_counts_subset),
                )
                self.data_frame_dict["subset_rmsd_df"].loc[
                    self.name_subset_list, dft_type
                ] = 100 * np.sqrt(
                    np.einsum(
                        "i,ij,j,j->j",
                        delta**2,
                        reactions_to_subset,
                        inverse_mae,
                        divide(1, completed_counts_subset),
                    )
                )
            elif self.data_set in DIET_DATASETS:
                wtmad_2_subset = (
                    1
                    / np.sum(total_counts_subset)
                    * np.einsum(
                        "i,i,i->i",
                        total_counts_subset,
                        weight_list,
                        mean_reaction_energy,
                    )
                )
                self.data_frame_dict["subset_mean_df"].loc[
                    self.name_subset_list, dft_type
                ] = mean_reaction_energy
                self._update_wtmad2_frames(dft_type, wtmad_2_subset, subset2set)

        if self.data_set == "gmtkn-def2":
            len_processed = np.sum(completed_counts_subset)
            log_dict = {
                "WTMAD-2_min": self.data_frame_dict["wtmad_2_df"].loc[
                    "summary", dft_type_list[0]
                ],
                "WTMAD-2_max": self.data_frame_dict["wtmad_2_df"].loc[
                    "summary", dft_type_list[0]
                ]
                / len_processed
                * 1505,
                "len_processed": len_processed,
            }
            for name_set in self.full_subset_dict:
                log_dict[f"WTMAD-2_{name_set}"] = float(
                    self.data_frame_dict["wtmad_2_df"].loc[name_set, dft_type_list[0]]
                )
            self.run.log(log_dict)

        for df in self.data_frame_dict.values():
            # rename columns
            df.rename(columns=RENAME_DICT, inplace=True)

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.float_format",
            "{:.2f}".format,
        ):
            for df_name in self.data_frame_name_list:
                log_print(self.data_frame_dict[df_name])

        log_print(f"Saving excel backup files with timestamp: {self.stamp}")
        for df_name, df in self.data_frame_dict.items():
            log_print(f"{df_name}:\n{df}\n")
            df.to_csv(f"validate_hkqai/csv_backup/{df_name}_{self.stamp}.csv")
            df.round(3).to_excel(
                f"validate_hkqai/excel_backup/{df_name}_{self.stamp}.xlsx"
            )
        done = status_summary == "DONE"
        output_log_str = "".join(output_log_list)
        self.if_done = done
        self.save_done_cache(done, output_log_str)
        return done

    def save_csv(self):
        self.data.to_csv(self.data_path, index=False)

    def load_csv(self):
        if not self.data_path.exists():
            print(f"CSV file {self.data_path} does not exist. No data loaded.")
            return False
        self.data = pd.read_csv(self.data_path)
        return True


def parse_time(time_str):
    unit_scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    unit = time_str[-1].lower()
    if unit not in unit_scale:
        raise ValueError(f"Unknown time unit: {unit}")
    return int(time_str[:-1]) * unit_scale[unit]


if __name__ == "__main__":
    # print pid for monitoring
    print(f"Process ID: {os.getpid()}")
    parser = argparse.ArgumentParser(description="Collect and process validation data.")
    parser.add_argument(
        "--model_load",
        type=str,
        nargs="?",
        const="",
        default="",
        help="Model load string. If not provided, defaults to an empty string, which mean from the bechmark DFT data. If provided, it should be a string that identifies the model to load.",
    )
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--basis", type=str, required=True)
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--frequency", type=str, default="10s")
    parser.add_argument("--max_checks", type=int, default=0)
    parser.add_argument(
        "--data_set", type=str, default="gmtkn-def2", choices=DATA_FRAME_NAMES
    )
    args = parser.parse_args()

    collector = Collect_info(
        model_load=args.model_load,
        epoch=args.epoch,
        basis=args.basis,
        verbose=args.verbose,
        data_set=args.data_set,
    )
    num_checks = 0

    while not collector.if_done:
        collector.reset()
        loaded = collector.load_csv()
        if not loaded:
            print("No data loaded. Collecting new data.")
            collector.aggregate_data()
            collector.add_d3bj_correction()
            collector.save_csv()
        collector.if_done = collector.get_wtmad_2()
        print("Waiting for new data...", flush=True)

        num_checks += 1
        if args.max_checks != -1 and num_checks >= args.max_checks:
            print("Maximum number of checks reached. Exiting.")
            break

        time.sleep(parse_time(args.frequency))

    print("All data processed. Exiting.")
