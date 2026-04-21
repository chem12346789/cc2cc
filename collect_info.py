from pathlib import Path
import json
import shutil
import time
from datetime import datetime
import argparse
import os

import pandas as pd
import numpy as np
import wandb

AU2KCALMOL = 627.5094740631


class Collect_info:
    def __init__(self, model_load, basis, verbose=4):
        self.model_load = model_load
        self.basis = basis
        self.verbose = verbose
        self.data_path = (
            Path("validate_hkqai")
            / f"ccdft_{self.basis}_{self.model_load}_gmtkn-def2.csv"
        )
        self.if_done = False

        experiment_dict = {
            "model_load": self.model_load,
            "basis": self.basis,
            "verbose": self.verbose,
            "pid": os.getpid(),
        }
        self.run = wandb.init(
            project="DFT2CC_validation",
            resume="allow",
            name=f"collect_{self.model_load}_{self.basis}",
            config=experiment_dict,
            allow_val_change=True,
        )

        print("Collect_info initialized with the following parameters:")
        for key, value in experiment_dict.items():
            print(f"{key}: {value}")
        print("")

        with open("jupyter-notebook/subset.json") as f:
            self.full_subset_dict = json.load(f)["full_subset_dict"]

        for name_set, subset_list_ in self.full_subset_dict.items():
            self.full_subset_dict[name_set] = np.sort(subset_list_).tolist()

        self.name_subset_list = [
            name_subset
            for subset in self.full_subset_dict.values()
            for name_subset in subset
        ]
        self.data = pd.DataFrame()

    def reset(self):
        self.data = pd.DataFrame()
        if self.verbose > 3:
            print("Data reset.")

    def aggregate_data(self):
        for name_subset in self.name_subset_list:
            data_path_list = list(
                Path("validate").glob(
                    f"*{self.basis}*{self.model_load}*{name_subset}.csv"
                )
            )
            if len(data_path_list) != 1:
                if self.verbose > 3:
                    print(
                        f"Warning: Expected 1 file for {name_subset} but found {len(data_path_list)}"
                    )
                continue
            else:
                source_data_path = data_path_list[0]

            with open(source_data_path, "r") as f:
                self.data = pd.concat([self.data, pd.read_csv(f)], ignore_index=True)

    def add_d3bj_correction(self):
        if "scf_d3bj_ene" in self.data.columns:
            if self.verbose > 3:
                print("D3BJ correction already added.")
            return

        data_test = pd.read_csv("validate_hkqai_done/ccdft_def2-QZVP__gmtkn-def2.csv")
        data_test_name = np.array(data_test["name"])

        while "name" not in self.data.columns:
            return

        for i, name in enumerate(self.data["name"]):
            if "_0_1_0.0000" not in name:
                name = name.replace(self.basis, "def2-QZVP_0_1_0.0000")
            else:
                name = name.replace(self.basis, "def2-QZVP")
            col = np.where(data_test_name == name)[0]
            b3lyp_ene = data_test.loc[col, "b3lyp_ene"].values[0]
            b3lyp_d3bj_ene = data_test.loc[col, "b3lyp-d3bj_ene"].values[0]
            self.data.loc[i, "scf_d3bj_ene"] = self.data.loc[i, "scf_ene"] + (
                b3lyp_d3bj_ene - b3lyp_ene
            )

    def get_wtmad_2(self):
        data_name_list = (
            self.data["name"].str.split(f"_{self.basis}", regex=False).str[0].to_numpy()
        )
        with open(
            "/home/chenzihao/workspace/cc2cc_test5/cc2cc/utils/gmtkn-def2.json"
        ) as f:
            GMNTK55_json = json.load(f)

        reference_energy = []
        molecules_to_reactions = []
        reactions_to_subset = []
        reaction_count_list = []

        for i_subset, subset_name in enumerate(self.name_subset_list):
            i_subset_name = "BH76" if subset_name == "BH76RC" else subset_name
            reaction_dict = GMNTK55_json[f"reaction-{subset_name}"]
            reaction_count_list.append(len(reaction_dict))

            for i_reaction in reaction_dict.values():
                systems_list = i_reaction["systems"]
                stoichiometry_list = i_reaction["stoichiometry"]
                molecule_stoichiometry = np.zeros(len(data_name_list))
                finished = True

                for i in range(len(systems_list)):
                    mole_name = f"{i_subset_name}-{systems_list[i]}"
                    stoichiometry = int(stoichiometry_list[i])

                    if mole_name in GMNTK55_json:
                        if isinstance(GMNTK55_json[mole_name], str):
                            mole_name = GMNTK55_json[mole_name]

                    col = np.where(data_name_list == mole_name)[0]
                    if col.size == 1:
                        molecule_stoichiometry[col[0]] = stoichiometry
                    else:
                        finished = False

                if finished:
                    reference_energy.append(i_reaction["reference"])
                    molecules_to_reactions.append(molecule_stoichiometry)
                    subset_index = np.zeros(len(self.name_subset_list))
                    subset_index[i_subset] = 1
                    reactions_to_subset.append(subset_index)

        subset2set = np.zeros((len(self.name_subset_list), len(self.full_subset_dict)))
        for i_subset, subset_name in enumerate(self.name_subset_list):
            for i_set, set_name in enumerate(self.full_subset_dict.keys()):
                if subset_name in self.full_subset_dict[set_name]:
                    subset2set[i_subset, i_set] = 1

        reference_energy = np.array(reference_energy)
        molecules_to_reactions = np.array(molecules_to_reactions)
        reactions_to_subset = np.array(reactions_to_subset)
        completed_reaction_counts = np.sum(reactions_to_subset, axis=0)
        reaction_count_list = np.array(reaction_count_list)

        print(f"Total number of reactions done: {np.sum(completed_reaction_counts)}")
        print(f"Total number of reactions: {np.sum(reaction_count_list)}")

        summary_subset_df = pd.read_csv(
            f"validate_hkqai_done/summary_subset.csv",
            index_col=0,
        )
        summary_wtmad_2_subset_df = pd.read_csv(
            f"validate_hkqai_done/summary_subset_wtmad_2.csv",
            index_col=0,
        )
        wtmad_1_df = pd.read_csv(
            f"validate_hkqai_done/wtmad_1_subset.csv",
            index_col=0,
        )
        wtmad_2_df = pd.read_csv(
            f"validate_hkqai_done/wtmad_2_subset.csv",
            index_col=0,
        )
        dft_type_list = ["scf_ene", "scf_d3bj_ene"]
        header = dft_type_list + ["Processed"]
        for df in [
            summary_subset_df,
            summary_wtmad_2_subset_df,
            wtmad_1_df,
            wtmad_2_df,
        ]:
            for col in header:
                if col not in df.columns:
                    df[col] = 0.0
            # move Processed column to the end
            df.pop("Processed")

        reaction_status_set = np.concatenate(
            [completed_reaction_counts[:, None], reaction_count_list[:, None]], axis=1
        )
        reaction_status_set = np.array(
            [
                f"{int(x[0])}/{int(x[1])}" if int(x[0]) < int(x[1]) else "DONE"
                for x in reaction_status_set
            ],
            dtype=object,
        )
        print(f"Number of reactions process/done in each subset: {reaction_status_set}")
        for df in (summary_subset_df, summary_wtmad_2_subset_df):
            df.loc[self.name_subset_list, "Processed"] = reaction_status_set

        for dft_type in dft_type_list:
            data_dft_ene = self.data[dft_type].to_numpy() * AU2KCALMOL

            # mae: mean_relative_abs_energies
            inverse_mae = 1 / np.einsum(
                "i,ij,j->j",
                np.abs(reference_energy),
                reactions_to_subset,
                1 / completed_reaction_counts,
            )
            inverse_mae[np.isnan(inverse_mae)] = 0

            reaction_energy_dft = np.einsum(
                "ji,i->j",
                molecules_to_reactions,
                data_dft_ene,
            )
            mean_absolute_deviation = 56.84 / 1505
            mean_reaction_energy = np.einsum(
                "i,ij,j->j",
                np.abs(reference_energy - reaction_energy_dft),
                reactions_to_subset,
                1 / completed_reaction_counts,
            )
            mean_reaction_energy[np.isnan(mean_reaction_energy)] = 0

            wtmad_1_multiplier = np.ones_like(completed_reaction_counts)
            wtmad_1_multiplier[completed_reaction_counts > 75] = 0.1
            wtmad_1_multiplier[completed_reaction_counts < 7.5] = 10
            wtmad_1_subset = wtmad_1_multiplier * mean_reaction_energy
            summary_subset_df.loc[self.name_subset_list, dft_type] = (
                mean_reaction_energy
            )

            wtmad_2_subset = mean_absolute_deviation * np.einsum(
                "i,i,i->i",
                completed_reaction_counts,
                inverse_mae,
                mean_reaction_energy,
            )
            summary_wtmad_2_subset_df.loc[self.name_subset_list, dft_type] = (
                wtmad_2_subset
            )
            print(
                f"WTMAD-2 for {dft_type}: {np.einsum('i,ij->j', wtmad_2_subset, subset2set)} sum: {np.sum(wtmad_2_subset)}"
            )

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(summary_subset_df)
            print(summary_wtmad_2_subset_df)

        raise NotImplementedError("WTMAD-2 calculation not implemented yet.")

        # len_processed = 0
        # for data_subset_i in data_subset.values():
        #     len_processed += len(data_subset_i["dft"])
        # len_processed = int(len_processed / len(dft_type_list))

        # for dft_type in dft_type_list:
        #     mean_absolute_deviation = 56.84 / 1505
        #     for name_set, subset_list_ in self.full_subset_dict.items():
        #         wtmad_1_dft = []
        #         wtmad_2_dft = []
        #         processed = []

        #         for i_subset in subset_list_:
        #             name_subset = f"{dft_type}_{i_subset}"

        #             if len(data_subset[name_subset]["dft"]) == 0:
        #                 summary_subset.loc[i_subset, dft_type] = 0
        #                 summary_subset.loc[i_subset, "Processed"] = (
        #                     f"0 / {len(data_subset[name_subset]['name'])}"
        #                 )
        #                 summary_wtmad_2_subset.loc[i_subset, dft_type] = 0
        #                 summary_wtmad_2_subset.loc[i_subset, "Processed"] = (
        #                     f"0 / {len(data_subset[name_subset]['name'])}"
        #                 )
        #             else:
        #                 summary_subset.loc[i_subset, dft_type] = np.mean(
        #                     data_subset[name_subset]["dft"]
        #                 )
        #                 summary_subset.loc[i_subset, "Processed"] = (
        #                     "DONE"
        #                     if (
        #                         len(data_subset[name_subset]["dft"])
        #                         == len(data_subset[name_subset]["name"])
        #                     )
        #                     else f"{len(data_subset[name_subset]['dft'])} / "
        #                     f"{len(data_subset[name_subset]['name'])}"
        #                 )
        #                 summary_wtmad_2_subset.loc[i_subset, dft_type] = (
        #                     len(data_subset[name_subset]["name"])
        #                     * mean_absolute_deviation
        #                     / np.mean(data_subset[name_subset]["cc"])
        #                     * np.mean(data_subset[name_subset]["dft"])
        #                 )
        #                 summary_wtmad_2_subset.loc[i_subset, "Processed"] = (
        #                     "DONE"
        #                     if (
        #                         len(data_subset[name_subset]["dft"])
        #                         == len(data_subset[name_subset]["name"])
        #                     )
        #                     else f"{len(data_subset[name_subset]['dft'])} / "
        #                     f"{len(data_subset[name_subset]['name'])}"
        #                 )

        #                 if np.mean(data_subset[name_subset]["cc"]) > 75:
        #                     wtmad_1 = 0.1
        #                 elif np.mean(data_subset[name_subset]["cc"]) < 7.5:
        #                     wtmad_1 = 10
        #                 else:
        #                     wtmad_1 = 1

        #                 wtmad_1_dft = np.append(
        #                     wtmad_1_dft,
        #                     wtmad_1 * np.mean(data_subset[name_subset]["dft"]) / 55,
        #                 )
        #                 wtmad_2_dft = np.append(
        #                     wtmad_2_dft,
        #                     len(data_subset[name_subset]["name"])
        #                     * mean_absolute_deviation
        #                     / np.mean(data_subset[name_subset]["cc"])
        #                     * np.mean(data_subset[name_subset]["dft"]),
        #                 )
        #             if (
        #                 len(data_subset[name_subset]["dft"])
        #                 == len(data_subset[name_subset]["name"])
        #             ) and len(data_subset[name_subset]["name"]) > 0:
        #                 processed.append(1)
        #             else:
        #                 processed.append(0)

        #         if len(wtmad_1_dft) == 0:
        #             wtmad_1_subset.loc[name_set, dft_type] = 0.0
        #         else:
        #             wtmad_1_subset.loc[name_set, dft_type] = np.sum(wtmad_1_dft)
        #         if len(wtmad_2_dft) == 0:
        #             wtmad_2_subset.loc[name_set, dft_type] = 0.0
        #         else:
        #             wtmad_2_subset.loc[name_set, dft_type] = np.sum(wtmad_2_dft)
        #         wtmad_1_subset.loc[name_set, "Processed"] = (
        #             f"{sum(processed)} / " f"{len(processed)}"
        #         )
        #         wtmad_2_subset.loc[name_set, "Processed"] = (
        #             f"{sum(processed)} / " f"{len(processed)}"
        #         )

        #     wtmad_1_subset.loc["summary", "Processed"] = "0/0"
        #     wtmad_2_subset.loc["summary", "Processed"] = "0/0"
        #     wtmad_1_subset.loc["summary", dft_type] = 0.0
        #     wtmad_2_subset.loc["summary", dft_type] = 0.0
        #     for name_set in self.full_subset_dict.keys():
        #         wtmad_1_subset.loc["summary", dft_type] += wtmad_1_subset.loc[
        #             name_set, dft_type
        #         ]
        #         wtmad_2_subset.loc["summary", dft_type] += wtmad_2_subset.loc[
        #             name_set, dft_type
        #         ]
        #         wtmad_1_subset.loc["summary", "Processed"] = (
        #             f"{int(wtmad_1_subset.loc["summary", "Processed"].split('/')[0]) + int(wtmad_1_subset.loc[name_set, 'Processed'].split('/')[0])} / "
        #             f"{int(wtmad_1_subset.loc["summary", "Processed"].split('/')[1]) + int(wtmad_1_subset.loc[name_set, 'Processed'].split('/')[1])}"
        #         )
        #         wtmad_2_subset.loc["summary", "Processed"] = (
        #             f"{int(wtmad_2_subset.loc["summary", "Processed"].split('/')[0]) + int(wtmad_2_subset.loc[name_set, 'Processed'].split('/')[0])} / "
        #             f"{int(wtmad_2_subset.loc["summary", "Processed"].split('/')[1]) + int(wtmad_2_subset.loc[name_set, 'Processed'].split('/')[1])}"
        #         )
        # wtmad_2_scf_ene_summary = wtmad_2_subset["scf_ene"].loc["summary"]

        # for name_set in list(self.full_subset_dict.keys()):
        #     if (
        #         wtmad_1_subset.loc[name_set, "Processed"].split("/")[0].strip()
        #         == wtmad_1_subset.loc[name_set, "Processed"].split("/")[1].strip()
        #     ):
        #         wtmad_1_subset.loc[name_set, "Processed"] = "DONE"
        #     if (
        #         wtmad_2_subset.loc[name_set, "Processed"].split("/")[0].strip()
        #         == wtmad_2_subset.loc[name_set, "Processed"].split("/")[1].strip()
        #     ):
        #         wtmad_2_subset.loc[name_set, "Processed"] = "DONE"

        # for name_set in ["summary"]:
        #     if (
        #         wtmad_1_subset.loc[name_set, "Processed"].split("/")[0].strip()
        #         == wtmad_1_subset.loc[name_set, "Processed"].split("/")[1].strip()
        #     ):
        #         wtmad_1_subset.loc[name_set, "Processed"] = "DONE"
        #         self.if_done = True
        #     if (
        #         wtmad_2_subset.loc[name_set, "Processed"].split("/")[0].strip()
        #         == wtmad_2_subset.loc[name_set, "Processed"].split("/")[1].strip()
        #     ):
        #         wtmad_2_subset.loc[name_set, "Processed"] = "DONE"
        #         self.if_done = True

        # print(
        #     f"{wtmad_2_scf_ene_summary:6.2f}"
        #     f"{wtmad_2_scf_ene_summary / len_processed * 1505:6.2f}"
        # )
        # print(len_processed)

        # print("WTMAD-2 Summary:")
        # with pd.option_context("display.max_rows", None, "display.max_columns", None):
        #     print(wtmad_2_subset)

        # log_dict = {
        #     "WTMAD-2_min": wtmad_2_scf_ene_summary,
        #     "WTMAD-2_max": wtmad_2_scf_ene_summary / len_processed * 1505,
        #     "len_processed": len_processed,
        # }
        # for name_set in self.full_subset_dict:
        #     log_dict[f"WTMAD-2_{name_set}"] = float(
        #         wtmad_2_subset.loc[name_set, "scf_ene"]
        #     )
        # self.run.log(log_dict)

        # date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        # # check if the df is the sota version by checking if the summary of scf_ene in wtmad_2_subset is smaller than the sota version
        # if (
        #     wtmad_2_subset.loc["summary", "scf_ene"]
        #     < wtmad_2_subset.loc["summary", "scf_ene_sota"]
        # ) and self.if_done:
        #     print("New sota achieved, updating sota backup files.")
        #     summary_subset_sota = summary_subset.copy()
        #     summary_wtmad_2_subset_sota = summary_wtmad_2_subset.copy()
        #     wtmad_1_subset_sota = wtmad_1_subset.copy()
        #     wtmad_2_subset_sota = wtmad_2_subset.copy()
        #     for df in [
        #         summary_subset_sota,
        #         summary_wtmad_2_subset_sota,
        #         wtmad_1_subset_sota,
        #         wtmad_2_subset_sota,
        #     ]:
        #         # delete the scf_ene_sota and scf_d3bj_ene_sota columns if exist in the dataframe
        #         for col in ["scf_ene_sota", "scf_d3bj_ene_sota"]:
        #             if col in df.columns:
        #                 df.drop(columns=[col], inplace=True)
        #         # rename the scf_ene and scf_d3bj_ene columns to include sota suffix
        #         if "scf_ene" in df.columns:
        #             df.rename(columns={"scf_ene": "scf_ene_sota"}, inplace=True)
        #         if "scf_d3bj_ene" in df.columns:
        #             df.rename(
        #                 columns={"scf_d3bj_ene": "scf_d3bj_ene_sota"}, inplace=True
        #             )
        #     print(f"Saving csv backup files with timestamp: {date}")
        #     summary_subset_sota.to_csv(f"validate_hkqai/csv_backup/summary_subset.csv")
        #     summary_wtmad_2_subset_sota.to_csv(
        #         f"validate_hkqai/csv_backup/summary_subset_wtmad_2.csv"
        #     )
        #     wtmad_1_subset_sota.to_csv(f"validate_hkqai/csv_backup/wtmad_1_subset.csv")
        #     wtmad_2_subset_sota.to_csv(f"validate_hkqai/csv_backup/wtmad_2_subset.csv")

        # for df in [
        #     summary_subset,
        #     summary_wtmad_2_subset,
        #     wtmad_1_subset,
        #     wtmad_2_subset,
        # ]:
        #     for col in df.columns:
        #         if col != "Processed":
        #             df[col] = df[col].map("{:.2f}".format)

        # print(f"Saving excel backup files with timestamp: {date}")
        # summary_subset.to_excel(
        #     f"validate_hkqai/excel_backup/summary_subset_{date}.xlsx"
        # )
        # summary_wtmad_2_subset.to_excel(
        #     f"validate_hkqai/excel_backup/summary_subset_wtmad_2_{date}.xlsx"
        # )
        # wtmad_1_subset.to_excel(
        #     f"validate_hkqai/excel_backup/wtmad_1_subset_{date}.xlsx"
        # )
        # wtmad_2_subset.to_excel(
        #     f"validate_hkqai/excel_backup/wtmad_2_subset_{date}.xlsx"
        # )

    def save_csv(self):
        self.data.to_csv(self.data_path, index=False)

    def load_csv(self):
        self.data = pd.read_csv(self.data_path)


def parse_time(time_str):
    unit = time_str[-1]
    value = int(time_str[:-1])
    if unit.lower() == "s":
        return value
    elif unit.lower() == "m":
        return value * 60
    elif unit.lower() == "h":
        return value * 3600
    elif unit.lower() == "d":
        return value * 86400
    else:
        raise ValueError(f"Unknown time unit: {unit}")


if __name__ == "__main__":
    # print pid for monitoring
    print(f"Process ID: {os.getpid()}")
    parser = argparse.ArgumentParser(description="Collect and process validation data.")
    parser.add_argument(
        "--model_load",
        type=str,
        required=True,
        help="Model load identifier.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        required=True,
        help="Basis set used in calculations.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=4,
        help="Verbosity level for logging.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="10s",
        help="Frequency for checking new data.",
    )
    parser.add_argument(
        "--max_checks",
        type=int,
        default=0,
        help="Maximum number of checks before exiting, -1 for infinite.",
    )
    args = parser.parse_args()

    collector = Collect_info(
        model_load=args.model_load,
        basis=args.basis,
        verbose=args.verbose,
    )
    num_checks = 0

    while not collector.if_done:
        collector.reset()
        # collector.aggregate_data()
        # collector.add_d3bj_correction()
        # collector.save_csv()
        collector.load_csv()
        collector.get_wtmad_2()
        print("Waiting for new data...", flush=True)

        num_checks += 1
        if args.max_checks != -1 and num_checks >= args.max_checks:
            print("Maximum number of checks reached. Exiting.")
            break

        time.sleep(
            parse_time(args.frequency)
        )  # Sleep for the specified duration before checking again

    print("All data processed. Exiting.")
