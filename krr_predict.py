import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cc2cc.utils import AU2KCALMOL, ARRAY_USE_MIDDLE

from krr import load_data, add_args, hash_value
from krr import KernelRidgeModified

DATA_PATH = Path("data/grids_dft")

parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

print("gamma:", args.gamma)
print("alpha:", args.alpha)
print("kernel:", args.kernel, flush=True)

(
    input_dict,
    output_dict,
    weights_dict,
    coords_dict,
    keys_list,
) = load_data(
    args.molecular_list,
    args.extend_atom,
    args.extend_xyz,
    args.distance_list,
)

krr = KernelRidgeModified(
    alpha=args.alpha,
    gamma=args.gamma,
    kernel_type="rbf",
)

# load the model
if args.load_number == -1:
    list_of_path = list(Path("data/save/").glob("dual_coef-final-*.npz"))
    load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
else:
    load_path = Path(f"data/save/dual_coef-final-{args.load_number}.npz")
print(f"Load the model: {load_path}", flush=True)
model = np.load(load_path, allow_pickle=True)
model_coef = model["dual_coef"].item()
model_para = model["model_para"].item()
hashtable = model["hashtable"]
csv_file = Path(f"data/save/{load_path.stem}.csv")
csv_data = {
    "index": [],
    "predict_error": [],
    "energy_correct": [],
    "predict_error_abs": [],
    "energy_correct_abs": [],
    "predict_error_sum": [],
    "energy_correct_sum": [],
    "predict_error_abs_sum": [],
    "energy_correct_abs_sum": [],
}
if csv_file.exists():
    print(f"Warning: {csv_file} exists.", flush=True)

model_coef_keys = list(model_coef.keys())
model_coef_keys_array = np.array(
    [[int(j) for j in i.split("_")] for i in model_coef_keys]
)
print(model_coef_keys_array)

x_all = {}
y_all = {}
w_all = {}
# x_neighborhood = {}
# y_neighborhood = {}
# w_neighborhood = {}
dual_coef = {}

for key in keys_list:
    print(f"Key: {key}", flush=True)
    for index_ in range(len(input_dict[key])):
        if index_ % (len(input_dict[key]) / 10) == 0:
            print(
                f"Processing: {index_ / (len(input_dict[key]) / 10) * 10:.1f}%",
                flush=True,
            )

        index_round = hash_value(
            input_dict[key][index_],
            hashtable=hashtable,
        )

        if index_round not in x_all:
            x_all[index_round] = [input_dict[key][index_]]
            y_all[index_round] = [output_dict[key][index_]]
            w_all[index_round] = [weights_dict[key][index_]]
        else:
            x_all[index_round].append(input_dict[key][index_])
            y_all[index_round].append(output_dict[key][index_])
            w_all[index_round].append(weights_dict[key][index_])

x_keys = list(x_all.keys())
for index_ in x_keys:
    x_all[index_] = np.array(x_all[index_])
    y_all[index_] = np.array(y_all[index_])
    w_all[index_] = np.array(w_all[index_])

predict_error_sum = 0
energy_correct_sum = 0
predict_error_abs_sum = 0
energy_correct_abs_sum = 0

for index_ in x_keys:
    if index_ in model_coef:
        index_search = index_
    else:
        index_search_index = np.argmin(
            np.linalg.norm(
                model_coef_keys_array - np.array([int(j) for j in index_.split("_")]),
                axis=1,
            )
        )
        index_search = list(model_coef_keys)[index_search_index]
        print(f"Index search: {index_}, found: {index_search}", flush=True)

    krr.dual_coef_ = model_coef[index_search]
    krr.x_fit = model_para["x_fit"][index_search]
    krr.alpha = model_para["alpha"][index_search]
    krr.gamma = model_para["gamma"][index_search]
    krr.kernel_type = model_para["kernel_type"][index_search]

    predict_error = np.sum(
        (y_all[index_] - krr.predict_data(x_all[index_]))
        * w_all[index_]
        * x_all[index_][:, ARRAY_USE_MIDDLE]
    )
    energy_correct = np.sum(
        y_all[index_] * w_all[index_] * x_all[index_][:, ARRAY_USE_MIDDLE]
    )
    predict_error_abs = np.sum(
        np.abs(
            (y_all[index_] - krr.predict_data(x_all[index_]))
            * w_all[index_]
            * x_all[index_][:, ARRAY_USE_MIDDLE]
        )
    )
    energy_correct_abs = np.sum(
        np.abs(y_all[index_] * w_all[index_] * x_all[index_][:, ARRAY_USE_MIDDLE])
    )

    predict_error_sum += predict_error
    energy_correct_sum += energy_correct
    predict_error_abs_sum += predict_error_abs
    energy_correct_abs_sum += energy_correct_abs

    csv_data["index"].append(index_)
    csv_data["predict_error"].append(AU2KCALMOL * predict_error)
    csv_data["energy_correct"].append(AU2KCALMOL * energy_correct)
    csv_data["predict_error_abs"].append(AU2KCALMOL * predict_error_abs)
    csv_data["energy_correct_abs"].append(AU2KCALMOL * energy_correct_abs)
    csv_data["predict_error_sum"].append(AU2KCALMOL * predict_error_sum)
    csv_data["energy_correct_sum"].append(AU2KCALMOL * energy_correct_sum)
    csv_data["predict_error_abs_sum"].append(AU2KCALMOL * predict_error_abs_sum)
    csv_data["energy_correct_abs_sum"].append(AU2KCALMOL * energy_correct_abs_sum)

print(
    "\nSummary of the model:\n",
    f"Energy correct sum: {AU2KCALMOL * energy_correct_sum} KCAL/MOL\n",
    f"Predict error sum: {AU2KCALMOL * predict_error_sum} KCAL/MOL\n",
    f"Energy correct abs sum: {AU2KCALMOL * energy_correct_abs_sum} KCAL/MOL\n",
    f"Predict error abs sum: {AU2KCALMOL * predict_error_abs_sum} KCAL/MOL\n",
    flush=True,
)

pd.DataFrame(csv_data).to_csv(csv_file, index=csv_data["index"])
